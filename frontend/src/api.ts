export type Position = [number, number];

export interface Feature<T = unknown> {
  type: "Feature";
  id?: string;
  geometry: { type: "Point" | "Polygon" | "MultiPolygon"; coordinates: unknown };
  properties: T;
}

export interface Collection<T = unknown> {
  type: "FeatureCollection";
  features: Feature<T>[];
  metadata: Record<string, unknown>;
}

export interface EventProperties {
  hotspot_count: number;
  first_seen: string;
  last_seen: string;
  bbox: [number, number, number, number];
  max_frp: number | null;
  mean_confidence: number | null;
  source_counts: Record<string, number>;
  hotspot_ids: string[];
  observation_zone: { type: "Polygon" | "MultiPolygon"; coordinates: unknown };
  observation_zone_area_ha: number;
  observation_zone_note: string;
  classification: "confirmed" | "candidate";
  quality_score: number;
}

export interface HotspotProperties {
  observed_at: string;
  source: string;
  confidence: string | number | null;
  frp: number | null;
}

export interface BurnProperties {
  area_ha?: number;
  forest_burned_ha?: number;
  severity?: "low" | "moderate" | "high";
}

export interface TimelineInterval {
  index: number;
  start_date: string;
  end_date: string;
  hotspots: Collection<HotspotProperties>;
}

export interface ObservationTimeline {
  bbox: number[];
  start_date: string;
  end_date: string;
  intervals: TimelineInterval[];
}

const baseUrl = import.meta.env.VITE_API_BASE_URL ?? "";
const responseCache = new Map<string, { expiresAt: number; value: unknown }>();

async function request<T>(endpoint: string, signal?: AbortSignal, ttlMs = 60_000): Promise<T> {
  const cacheKey = `${baseUrl}/api/v1${endpoint}`;
  const cached = responseCache.get(cacheKey);
  if (cached && cached.expiresAt > Date.now()) return cached.value as T;
  const response = await fetch(`${baseUrl}/api/v1${endpoint}`, { signal });
  if (!response.ok) {
    const body = await response.json().catch(() => null) as { detail?: string } | null;
    throw new Error(body?.detail ?? `Ошибка запроса (${response.status})`);
  }
  const value = await response.json() as T;
  responseCache.set(cacheKey, { value, expiresAt: Date.now() + ttlMs });
  return value;
}

const params = (values: Record<string, string | number>) => new URLSearchParams(
  Object.entries(values).map(([key, value]) => [key, String(value)]),
).toString();

export const api = {
  observationTimeline: (bbox: string, start_date: string, end_date: string, signal?: AbortSignal) =>
    request<ObservationTimeline>(`/observation-timeline?${params({ bbox, start_date, end_date })}`, signal, 60_000),
  hotspots: (bbox: string, days: number, startDate?: string, signal?: AbortSignal) =>
    request<Collection<HotspotProperties>>(`/hotspots?${params({ bbox, days, ...(startDate ? { start_date: startDate } : {}) })}`, signal, 60_000),
  events: (bbox: string, days: number, sensitivity: "strict" | "normal" | "aggressive", startDate?: string, signal?: AbortSignal) =>
    request<Collection<EventProperties>>(`/fire-events?${params({ bbox, days, sensitivity, ...(startDate ? { start_date: startDate } : {}) })}`, signal, 60_000),
  burnedAreas: (bbox: string, date: string, signal?: AbortSignal) =>
    request<Collection<BurnProperties>>(`/burned-areas?${params({ bbox, date })}`, signal, 3_600_000),
  severity: (bbox: string, before_date: string, after_date: string, signal?: AbortSignal) =>
    request<Collection<BurnProperties>>(`/severity/dnbr?${params({ bbox, before_date, after_date })}`, signal, 3_600_000),
  // Kept behind a semantic name so the UI is independent of the legacy route.
  bsMask: (bbox: string, before_date: string, after_date: string, signal?: AbortSignal) =>
    request<Collection<BurnProperties>>(`/severity/bs?${params({ bbox, before_date, after_date })}`, signal, 3_600_000),
  forestImpact: (bbox: string, before_date: string, after_date: string, signal?: AbortSignal) =>
    request<Collection<BurnProperties>>(`/forest-impact?${params({ bbox, before_date, after_date })}`, signal, 3_600_000),
  scenePreviewUrl: (bbox: string, startDate: string, endDate: string, position: "start" | "end") =>
    `${baseUrl}/api/v1/scene-preview?${params({ bbox, date: startDate, end_date: endDate, position })}`,
};
