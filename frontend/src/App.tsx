import { useEffect, useRef, useState } from "react";
import L from "leaflet";
import { api, type Collection, type BurnProperties, type HotspotProperties, type ObservationTimeline } from "./api";

type LoadState = "idle" | "loading" | "ready" | "empty" | "error";
type ViewMode = "live" | "archive";
const DEFAULT_BBOX = "92.5,55.8,93.2,56.4";
const day = (offset: number) => new Date(Date.now() + offset * 86_400_000).toISOString().slice(0, 10);
const normalizeBbox = (bbox: string): string | null => {
  const v = bbox.split(",").map(Number);
  if (v.length !== 4 || v.some((value) => !Number.isFinite(value))) return null;
  let [west, south, east, north] = v;
  if (south < -90 || north > 90 || south >= north || west >= east || east - west >= 360) return null;
  const worldShift = Math.floor(((west + east) / 2 + 180) / 360) * 360;
  west -= worldShift; east -= worldShift;
  if (west < -180 || east > 180) return null;
  return [west, south, east, north].map((value) => value.toFixed(5)).join(",");
};
const bounds = (bbox: string): L.LatLngBoundsExpression | null => { const normalized = normalizeBbox(bbox); if (!normalized) return null; const [west, south, east, north] = normalized.split(",").map(Number); return [[south, west], [north, east]]; };
const bboxFromBounds = (value: L.LatLngBounds) => { const sw = value.getSouthWest(), ne = value.getNorthEast(); return normalizeBbox([sw.lng, sw.lat, ne.lng, ne.lat].join(",")); };
const errorText = (error: unknown) => { const text = error instanceof Error ? error.message : "Не удалось получить данные"; return text.includes("422") ? "Проверьте границы и период (минимум 10 дней)." : text.includes("502") || text.includes("503") ? "Сервис NASA или спутниковых снимков временно недоступен." : text; };

export function App() {
  const mapElement = useRef<HTMLDivElement>(null), map = useRef<L.Map | null>(null), overlays = useRef<L.LayerGroup[]>([]), sceneOverlays = useRef(new Map<string, L.ImageOverlay>()), request = useRef<AbortController | null>(null);
  const [bbox, setBbox] = useState(DEFAULT_BBOX), [startDate, setStartDate] = useState(day(-30)), [endDate, setEndDate] = useState(day(-1));
  const [mode, setMode] = useState<ViewMode>("live"), [liveHotspots, setLiveHotspots] = useState<Collection<HotspotProperties> | null>(null), [liveState, setLiveState] = useState<LoadState>("idle");
  const [timeline, setTimeline] = useState<ObservationTimeline | null>(null), [analysisBbox, setAnalysisBbox] = useState<string | null>(null), [bsMask, setBsMask] = useState<Collection<BurnProperties> | null>(null), [position, setPosition] = useState(0), [timelineState, setTimelineState] = useState<LoadState>("idle"), [sceneState, setSceneState] = useState<LoadState>("idle"), [bsState, setBsState] = useState<LoadState>("idle"), [message, setMessage] = useState("Выберите режим и область наблюдения."), [showSatellite, setShowSatellite] = useState(true), [showHotspots, setShowHotspots] = useState(true), [showBs, setShowBs] = useState(true);
  const clearMap = () => { overlays.current.forEach((layer) => layer.remove()); overlays.current = []; };
  useEffect(() => { if (!mapElement.current || map.current) return; const instance = L.map(mapElement.current, { zoomControl: false, attributionControl: false, maxBounds: L.latLngBounds([-85.0511, -180], [85.0511, 180]), maxBoundsViscosity: 1, worldCopyJump: false }).fitBounds(bounds(DEFAULT_BBOX)!); instance.createPane("satellite-scene").style.zIndex = "250"; instance.createPane("bs-mask").style.zIndex = "500"; const syncBbox = () => { const next = bboxFromBounds(instance.getBounds()); if (next) setBbox(next); }; instance.on("moveend", syncBbox); L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 18, attribution: "© OpenStreetMap" }).addTo(instance); map.current = instance; return () => { instance.off("moveend", syncBbox); instance.remove(); map.current = null; }; }, []);
  const loadPeriod = async () => {
    if (startDate >= endDate) { setMessage("Дата начала должна быть раньше даты окончания."); return; }
    const normalizedBbox = normalizeBbox(bbox);
    if (!normalizedBbox || !bounds(normalizedBbox)) { setMessage("Введите границы в формате west,south,east,north с корректным порядком координат."); return; }
    if (normalizedBbox !== bbox) setBbox(normalizedBbox);
    request.current?.abort(); const controller = new AbortController(); request.current = controller;
    clearMap(); sceneOverlays.current.forEach((layer) => layer.remove()); sceneOverlays.current.clear(); setLiveHotspots(null); setLiveState("idle"); setTimeline(null); setAnalysisBbox(null); setBsMask(null); setPosition(0); setTimelineState("loading"); setSceneState("loading"); setBsState(showBs ? "loading" : "idle"); setMessage("Получаем 10 интервалов NASA FIRMS и спутниковые снимки…");
    try {
      const data = await api.observationTimeline(normalizedBbox, startDate, endDate, controller.signal); if (controller.signal.aborted) return;
      const targetBounds = bounds(normalizedBbox); setAnalysisBbox(normalizedBbox); setTimeline(data); setTimelineState(data.intervals.some((item) => item.hotspots.features.length) ? "ready" : "empty"); if (targetBounds) map.current?.fitBounds(targetBounds, { padding: [30, 30] }); setMessage("Период загружен. Перемещайте шкалу, чтобы просмотреть очаги по интервалам.");
      if (showBs) try { const mask = await api.bsMask(normalizedBbox, startDate, endDate, controller.signal); if (!controller.signal.aborted) { setBsMask(mask); setBsState(mask.features.length ? "ready" : "empty"); const resolution = Number(mask.metadata.resolution_m); if (Number.isFinite(resolution) && resolution > 20) setMessage(`Данные FIRMS готовы. Для большой области BS-маска рассчитана с адаптивным разрешением ${resolution.toFixed(1)} м.`); } } catch (error) { if (!controller.signal.aborted) { setBsState("error"); setMessage(`Данные FIRMS готовы. Маска BS: ${errorText(error)}`); } }
    } catch (error) { if (!controller.signal.aborted) { setTimelineState("error"); setSceneState("error"); setBsState(showBs ? "error" : "idle"); setMessage(errorText(error)); } }
  };
  const loadLive = async () => {
    const normalizedBbox = normalizeBbox(bbox);
    if (!normalizedBbox) { setMessage("Введите границы в формате west,south,east,north с корректным порядком координат."); return; }
    request.current?.abort(); const controller = new AbortController(); request.current = controller;
    clearMap(); sceneOverlays.current.forEach((layer) => layer.remove()); sceneOverlays.current.clear(); setTimeline(null); setBsMask(null); setAnalysisBbox(null); setTimelineState("idle"); setSceneState("idle"); setBsState("idle"); setLiveHotspots(null); setLiveState("loading"); setMessage("Получаем оперативные точки NASA FIRMS…");
    try { const hotspots = await api.hotspots(normalizedBbox, 1, undefined, controller.signal); if (controller.signal.aborted) return; setBbox(normalizedBbox); setLiveHotspots(hotspots); setLiveState(hotspots.features.length ? "ready" : "empty"); const targetBounds = bounds(normalizedBbox); if (targetBounds) map.current?.fitBounds(targetBounds, { padding: [30, 30] }); setMessage(hotspots.features.length ? `Загружено точек NASA FIRMS: ${hotspots.features.length}.` : "Оперативных точек FIRMS в этой области не найдено."); } catch (error) { if (!controller.signal.aborted) { setLiveState("error"); setMessage(errorText(error)); } }
  };
  const changeMode = (nextMode: ViewMode) => { if (nextMode === mode) return; request.current?.abort(); clearMap(); sceneOverlays.current.forEach((layer) => layer.remove()); sceneOverlays.current.clear(); setMode(nextMode); setLiveHotspots(null); setLiveState("idle"); setTimeline(null); setAnalysisBbox(null); setBsMask(null); setTimelineState("idle"); setSceneState("idle"); setBsState("idle"); setMessage(nextMode === "live" ? "Укажите область и загрузите оперативные точки FIRMS." : "Укажите период и загрузите архивные сведения."); };
  // Loading is intentionally user-initiated: opening the application must not
  // start costly FIRMS, satellite-scene, or BS-model requests.
  useEffect(() => () => { request.current?.abort(); sceneOverlays.current.forEach((layer) => layer.remove()); }, []);
  useEffect(() => {
    const currentMap = map.current;
    const sceneDate = timeline && analysisBbox && (position === 0 || position === 9) ? (position === 0 ? timeline.start_date : timeline.end_date) : null;
    sceneOverlays.current.forEach((layer) => layer.setOpacity(0));
    if (!currentMap || !sceneDate || !analysisBbox || !timeline || !showSatellite) { setSceneState("idle"); return; }
    const key = `${analysisBbox}:${timeline.start_date}:${timeline.end_date}:${position}`;
    const cached = sceneOverlays.current.get(key);
    const overlayBounds = bounds(analysisBbox); if (!overlayBounds) return;
    const overlay = cached ?? L.imageOverlay(api.scenePreviewUrl(analysisBbox, timeline.start_date, timeline.end_date, position === 0 ? "start" : "end"), overlayBounds, { opacity: 1, interactive: false, pane: "satellite-scene" });
    const image = overlay.getElement();
    overlay.setOpacity(1);
    if (image?.complete && image.naturalWidth > 0) { setSceneState("ready"); return; }
    setSceneState("loading");
    const onLoad = () => setSceneState("ready");
    const onError = () => {
      overlay.remove(); sceneOverlays.current.delete(key);
      setSceneState("error"); setMessage("Не удалось загрузить спутниковый снимок. Повторите запрос.");
    };
    overlay.once("load", onLoad);
    overlay.once("error", onError);
    sceneOverlays.current.set(key, overlay);
    if (!cached) overlay.addTo(currentMap);
    else if (image?.complete && image.naturalWidth === 0) onError();
    return () => { overlay.off("load", onLoad); overlay.off("error", onError); };
  }, [timeline, position, analysisBbox, showSatellite]);
  useEffect(() => { clearMap(); const current = map.current, interval = timeline?.intervals[position], hotspots = mode === "live" ? liveHotspots?.features : interval?.hotspots.features; if (!current || !hotspots) return; const layer = L.layerGroup().addTo(current); overlays.current.push(layer); if (showHotspots) hotspots.forEach((feature) => { const [lng, lat] = feature.geometry.coordinates as [number, number], p = feature.properties as HotspotProperties; L.circleMarker([lat, lng], { radius: Math.max(5, Math.min(12, 5 + (p.frp ?? 0) / 9)), color: "#fff4c7", weight: 1.5, fillColor: "#f05231", fillOpacity: .9 }).bindTooltip(`NASA FIRMS · ${p.source}<br>FRP: ${p.frp ?? "—"}`).addTo(layer); }); if (mode === "archive" && position === 9 && showBs && bsMask) { const colors = { low: "#f5dd49", moderate: "#f28c28", high: "#d94a3a" }; L.geoJSON(bsMask as unknown as GeoJSON.FeatureCollection, { pane: "bs-mask", style: (feature) => { const severity = (feature?.properties as BurnProperties | undefined)?.severity ?? "low"; const color = colors[severity]; return { color, weight: 2, fillColor: color, fillOpacity: .48 }; } }).bindTooltip("Маска сегментации BS-модели").addTo(layer); } }, [mode, liveHotspots, timeline, position, bsMask, showBs, showHotspots]);
  const interval = timeline?.intervals[position], status = (s: LoadState) => s === "loading" ? "загрузка…" : s === "error" ? "ошибка" : s === "empty" ? "нет точек" : "";
  return <main className={`app-shell ${timelineState === "idle" ? "is-idle" : ""}`}>
    <aside className="sidebar"><div className="brand"><span className="brand-mark" role="img" aria-label="Пламя"><svg viewBox="0 0 32 32" aria-hidden="true"><path className="flame-outer" d="M17.4 2.8c.8 5.4-3.4 7.1-3.2 11.1-2.1-1.3-2.6-3.3-2.2-5.5C7.8 11.7 5.4 15.2 5.8 20c.5 6.1 5 9.2 10.2 9.2 5.8 0 10.4-3.6 10.2-9.5-.2-5.2-3.7-9.7-8.8-16.9Z"/><path className="flame-inner" d="M17.1 15.1c.2 2.5-2 3.4-1.7 5.4-1.1-.7-1.5-1.8-1.3-3-2 1.8-2.8 3.6-2.5 5.6.3 2.6 2.2 4.1 4.7 4.1 2.7 0 4.8-1.7 4.7-4.5-.1-2.5-1.6-4.7-3.9-7.6Z"/></svg></span><div><strong>Лесной дозор</strong><small>Наблюдение за периодом</small></div></div>
      <div className="mode-switch" role="group" aria-label="Режим работы"><button className={mode === "live" ? "active" : ""} onClick={() => changeMode("live")}>Оперативный режим</button><button className={mode === "archive" ? "active" : ""} onClick={() => changeMode("archive")}>Просмотр архивных сведений</button></div>
      <div className="sidebar-content">
      {mode === "live" ? <section className="control-group"><div className="section-title">Оперативный поиск</div><label>Границы области<input value={bbox} onChange={(e) => setBbox(e.target.value)} onBlur={() => { const enteredBounds = bounds(bbox); if (enteredBounds) map.current?.fitBounds(enteredBounds, { padding: [30, 30] }); else setMessage("Введите границы в формате west,south,east,north с корректным порядком координат."); }} /></label><button className="primary" onClick={() => void loadLive()} disabled={liveState === "loading"}>↻ {liveState === "loading" ? "Загружаем…" : "Загрузить"}</button></section> : <>
      <section className="control-group"><div className="section-title">Область и период</div><label>Границы области<input value={bbox} onChange={(e) => setBbox(e.target.value)} onBlur={() => { const enteredBounds = bounds(bbox); if (enteredBounds) map.current?.fitBounds(enteredBounds, { padding: [30, 30] }); else setMessage("Введите границы в формате west,south,east,north с корректным порядком координат."); }} /></label><label>Дата начала периода<input type="date" value={startDate} max={endDate} onChange={(e) => setStartDate(e.target.value)} /></label><label>Дата окончания периода<input type="date" value={endDate} min={startDate} onChange={(e) => setEndDate(e.target.value)} /></label><button className="primary" onClick={() => void loadPeriod()} disabled={timelineState === "loading"}>↻ {timelineState === "loading" ? "Загружаем…" : "Загрузить период"}</button></section>
      <section className="control-group layers"><div className="section-title">Комбинация карт</div>
        <label className={`layer ${showSatellite ? "is-active" : ""}`}><input type="checkbox" checked={showSatellite} onChange={(e) => setShowSatellite(e.target.checked)} /><span className="layer-radio" /><span className="swatch satellite" /><span className="layer-label">Спутниковый снимок</span><em>{status(sceneState)}</em></label>
        <label className={`layer ${showHotspots ? "is-active" : ""}`}><input type="checkbox" checked={showHotspots} onChange={(e) => setShowHotspots(e.target.checked)} /><span className="layer-radio" /><span className="swatch" /><span className="layer-label">Точки NASA FIRMS</span><em>{status(timelineState)}</em></label>
        <label className={`layer ${showBs ? "is-active" : ""}`}><input type="checkbox" checked={showBs} onChange={(e) => setShowBs(e.target.checked)} /><span className="layer-radio" /><span className="swatch severity" /><span className="layer-label">Изменения dNBR <small>(BS-модель)</small></span><em>{status(bsState)}</em></label>
      </section><div className="notice"><span>i</span><p>{message}</p></div><footer>Снимки выбираются из дневных наблюдений с минимальной облачностью.</footer></>}
      </div>
    </aside>
    <section className="map-area"><div ref={mapElement} className="map" />{sceneState === "loading" ? <div className="map-loader">Загружаем спутниковый слой…</div> : null}{position === 9 && showBs && bsState === "ready" ? <div className="bs-legend"><b>BS-маска</b><span><i className="low" />низкая</span><span><i className="moderate" />средняя</span><span><i className="high" />высокая</span></div> : null}<section className="timeline"><div className="timeline-head"><div><b>Временная шкала NASA FIRMS</b><span>{interval ? `${interval.start_date} — ${interval.end_date}` : "Ожидание данных"}</span></div><em>{status(timelineState)}</em></div><input type="range" min="0" max="9" value={position} disabled={!timeline || timelineState === "loading"} onChange={(e) => setPosition(Number(e.target.value))} /><div className="ticks">{timeline?.intervals.map((item) => <button className={item.index === position ? "active" : ""} onClick={() => setPosition(item.index)} key={item.index} disabled={timelineState === "loading"}>{item.index + 1}<small>{item.hotspots.features.length}</small></button>)}</div>{timelineState === "error" || sceneState === "error" || bsState === "error" ? <button className="retry" onClick={() => void loadPeriod()}>Повторить запрос</button> : null}</section></section>
  </main>;
}
