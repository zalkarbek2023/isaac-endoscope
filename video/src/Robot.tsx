// "A robot endoscope finds polyps by itself": top = real RTX frames from NVIDIA Isaac Sim with the TensorRT detector's
// box and the robot's mode; bottom = x-ray of the colon digital twin with the scope's path. 4:5, 30 fps.
import React, {useMemo} from 'react';
import {AbsoluteFill, Img, interpolate, staticFile, useCurrentFrame, Easing} from 'remotion';
import {ThreeCanvas} from '@remotion/three';
import {useThree} from '@react-three/fiber';
import * as THREE from 'three';
import {loadFont as loadOnest} from '@remotion/google-fonts/Onest';
import {loadFont as loadMono} from '@remotion/google-fonts/JetBrainsMono';
import twin from '../public/twin.json';

const {fontFamily: SANS} = loadOnest('normal', {weights: ['400', '600', '800'], subsets: ['latin']});
const {fontFamily: MONO} = loadMono('normal', {weights: ['400', '500'], subsets: ['latin']});

type P = {t: number; p: number[]; s: 'N' | 'A' | 'I'; c: number; b: number[] | null};
type Find = {t: number; lesion: string; class: string; size_mm: number};
const T = twin as unknown as {wall: {v: number[][]; f: number[][]}; center: number[][];
  polyps: {id: string; class: string; size_mm: number; top: number[]; v: number[][]; f: number[][]}[];
  path: P[]; summary: {finds: Find[]; found_ids: string[]; lesions: string[]; seconds: number; steps: number; false_alarms?: number}};

export const W = 1080, H = 1350, FPS = 30;
const INTRO = 3.5, MAIN = 26, OUTRO = 6;
export const TOTAL = Math.round((INTRO + MAIN + OUTRO) * FPS);
const N = T.path.length;
const TOP_H = 864;                                               // 640x512 RTX frames scaled to 1080 wide
const BG = '#05070C', TEXT = '#F4F1EA', DIM = '#8A929C', NV = '#76B900', AMBER = '#FFB020';
const NAME: Record<string, string> = {AD: 'adenoma', HP: 'hyperplastic', SSL: 'serrated lesion', TSA: 'serrated adenoma'};
const MODE: Record<string, [string, string]> = {N: ['NAVIGATE', '#7DB7FF'], A: ['APPROACH', AMBER], I: ['INSPECT', NV]};
const CLAMP = {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'} as const;
const OUT = Easing.bezier(0.16, 1, 0.3, 1);
const ramp = (f: number, a: number, b: number) => interpolate(f, [a * FPS, b * FPS], [0, 1], {...CLAMP, easing: OUT});

// time remap: approach/inspect play in real time, cruising is fast-forwarded so the whole run fits MAIN seconds
// weights (video frames per run frame): inspect real-time, approach ~2x, cruising whatever is left so the run fits
const INTRO_RUN = Math.floor(INTRO * FPS * 0.5);                 // intro: first seconds of the run at half speed
const body = T.path.slice(INTRO_RUN);
const nI = body.filter((p) => p.s === 'I').length, nA = body.filter((p) => p.s === 'A').length, nN = body.length - nI - nA;
let wA = 0.5, wN = (MAIN * FPS - nI - nA * wA) / Math.max(1, nN);
if (wN < 0.12) { wN = 0.12; wA = Math.max(0.1, (MAIN * FPS - nI - nN * wN) / Math.max(1, nA)); }
const WEIGHT = {I: 1, A: wA, N: wN} as const;
const CUM: number[] = [];                                       // video-frame offset (in MAIN) where each run frame starts
{ let acc = 0; for (let k = 0; k < N; k++) { CUM.push(acc); if (k >= INTRO_RUN) acc += WEIGHT[T.path[k].s]; } }
const runIdx = (f: number) => {
  if (f < INTRO * FPS) return Math.min(N - 1, Math.floor(f * 0.5));
  const v = f - INTRO * FPS;
  let lo = INTRO_RUN, hi = N - 1;                               // last run frame whose start <= v
  while (lo < hi) { const m = (lo + hi + 1) >> 1; if (CUM[m] <= v) lo = m; else hi = m - 1; }
  return lo;
};
// honest numbers from all evaluation runs (eval_runs.sh, detector polyp_sim.engine)
const EVAL = {runs: 3, all3: 3, alarms: 10, falseAlarms: 1, controlFalse: 2, controlS: 32};

const Inside: React.FC = () => {
  const f = useCurrentFrame();
  const i = runIdx(f);
  const p = T.path[i];
  const k = W / 640;
  const [mode, col] = MODE[p.s];
  const showBox = p.b && (p.c >= 0.45 || p.s !== 'N');
  return (
    <div style={{position: 'absolute', left: 0, top: 0, width: W, height: TOP_H, overflow: 'hidden'}}>
      <Img src={staticFile(`frames/${String(i).padStart(5, '0')}.jpg`)} style={{width: W, height: TOP_H}} />
      {showBox && p.b && (
        <div style={{position: 'absolute', left: p.b[0] * k, top: p.b[1] * k, width: (p.b[2] - p.b[0]) * k, height: (p.b[3] - p.b[1]) * k,
          border: `5px solid ${col}`, borderRadius: 14, boxShadow: `0 0 26px ${col}`}}>
          <div style={{position: 'absolute', left: -5, top: -48, padding: '5px 12px', borderRadius: 9, background: col, color: '#071000',
            fontFamily: MONO, fontWeight: 500, fontSize: 24, whiteSpace: 'nowrap'}}>polyp · {p.c.toFixed(2)}</div>
        </div>
      )}
      <div style={{position: 'absolute', left: 32, bottom: 26, display: 'flex', gap: 12, fontFamily: MONO, fontSize: 24}}>
        <span style={{padding: '8px 16px', borderRadius: 30, background: 'rgba(5,7,12,0.78)', color: col, border: `2px solid ${col}`}}>● {mode}</span>
        <span style={{padding: '8px 16px', borderRadius: 30, background: 'rgba(5,7,12,0.78)', color: TEXT}}>t = {p.t.toFixed(1)} s</span>
      </div>
      <div style={{position: 'absolute', right: 32, bottom: 26, padding: '8px 16px', borderRadius: 30, background: 'rgba(5,7,12,0.78)',
        fontFamily: MONO, fontSize: 22, color: DIM}}>Isaac Sim RTX · TensorRT</div>
    </div>
  );
};

// ---------- x-ray of the digital twin ----------
function geo(v: number[][], fi: number[][]) {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(v.flat(), 3));
  g.setIndex(fi.flat());
  g.computeVertexNormals();
  return g;
}

const Rig: React.FC<{f: number; tip: THREE.Vector3}> = ({f, tip}) => {
  const {camera} = useThree();
  const a = -0.9 + f / FPS * 0.035;
  const look = new THREE.Vector3(0, 0, 20).lerp(tip, 0.55);
  camera.position.set(look.x + Math.cos(a) * 30, look.y + 11, look.z + Math.sin(a) * 30);
  camera.lookAt(look);
  return null;
};

const XRay: React.FC = () => {
  const f = useCurrentFrame();
  const i = runIdx(f);
  const wall = useMemo(() => geo(T.wall.v, T.wall.f), []);
  const polyps = useMemo(() => T.polyps.map((p) => geo(p.v, p.f)), []);
  const pts = useMemo(() => {
    const raw = T.path.map((p) => new THREE.Vector3(...(p.p as [number, number, number])));
    return raw.map((_, k) => {                                   // moving average: hide the wall-sweep wiggle in the x-ray
      const a = Math.max(0, k - 45), b = Math.min(raw.length - 1, k + 45), m = new THREE.Vector3();   // one sweep period = 90 frames
      for (let j = a; j <= b; j++) m.add(raw[j]);
      return m.multiplyScalar(1 / (b - a + 1));
    });
  }, []);
  const tip = pts[i];
  const body = useMemo(() => new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts.slice(Math.max(0, i - 400), i + 1).filter((_, k, arr) => k % 4 === 0 || k === arr.length - 1).length > 1
    ? pts.slice(Math.max(0, i - 400), i + 1).filter((_, k, arr) => k % 4 === 0 || k === arr.length - 1) : [pts[0], pts[1]]), 120, 0.32, 10, false), [i, pts]);
  const t = T.path[i].t;
  return (
    <ThreeCanvas width={W} height={H - TOP_H} camera={{fov: 32, near: 0.1, far: 500}} gl={{antialias: true, preserveDrawingBuffer: true}}
      style={{position: 'absolute', left: 0, top: TOP_H}}>
      <color attach="background" args={[BG]} />
      <ambientLight intensity={0.6} />
      <directionalLight position={[20, 40, 10]} intensity={1.4} />
      <Rig f={f} tip={tip} />
      <mesh geometry={wall}>
        <meshBasicMaterial color="#ff7a55" transparent opacity={0.07} side={THREE.BackSide} depthWrite={false} blending={THREE.AdditiveBlending} />
      </mesh>
      <mesh geometry={wall}>
        <meshBasicMaterial color="#ffb59a" wireframe transparent opacity={0.09} depthWrite={false} blending={THREE.AdditiveBlending} />
      </mesh>
      {T.polyps.map((p, k) => {
        const found = T.summary.finds.some((x) => x.lesion === p.id && x.t <= t);
        return (
          <group key={p.id}>
            <mesh geometry={polyps[k]}>
              <meshBasicMaterial color={found ? NV : AMBER} />
            </mesh>
            <mesh position={p.top as [number, number, number]}>
              <sphereGeometry args={[found ? 1.1 : 0.8, 20, 20]} />
              <meshBasicMaterial color={found ? NV : AMBER} transparent opacity={found ? 0.35 : 0.22} depthWrite={false} blending={THREE.AdditiveBlending} />
            </mesh>
          </group>
        );
      })}
      <mesh geometry={body}>
        <meshStandardMaterial color="#cfd6de" metalness={0.6} roughness={0.35} />
      </mesh>
      <mesh position={tip}>
        <sphereGeometry args={[0.5, 16, 16]} />
        <meshBasicMaterial color={MODE[T.path[i].s][1]} />
      </mesh>
      <pointLight position={tip} color="#ffffff" intensity={6} distance={6} />
    </ThreeCanvas>
  );
};

const FoundCard: React.FC = () => {
  const f = useCurrentFrame();
  const t = T.path[runIdx(f)].t;
  const hit = [...T.summary.finds].reverse().find((x) => t >= x.t && t < x.t + 3.0);   // inspect is real-time, so ~3 s on screen
  if (!hit || f / FPS < INTRO) return null;
  return (
    <div style={{position: 'absolute', left: 40, right: 40, top: 36, padding: '22px 30px', borderRadius: 26, background: 'rgba(5,7,12,0.84)',
      border: `2px solid ${NV}`, fontFamily: SANS, color: TEXT, boxShadow: `0 0 40px rgba(118,185,0,0.35)`}}>
      <div style={{fontFamily: MONO, fontSize: 24, color: NV}}>✓ POLYP FOUND · {hit.t.toFixed(1)} s</div>
      <div style={{fontSize: 52, fontWeight: 800, letterSpacing: -1.5, marginTop: 6}}>{hit.size_mm} mm {NAME[hit.class] ?? hit.class}</div>
    </div>
  );
};

const Intro: React.FC = () => {
  const f = useCurrentFrame();
  const o = ramp(f, 0.2, 0.8) * (1 - ramp(f, INTRO - 0.4, INTRO));
  if (o < 0.01) return null;
  return (
    <div style={{position: 'absolute', left: 48, right: 48, top: 40, opacity: o, fontFamily: SANS, fontWeight: 800, fontSize: 84, lineHeight: 1.02,
      letterSpacing: -3, color: TEXT, textShadow: '0 4px 30px rgba(0,0,0,0.8)'}}>
      A robot endoscope<div style={{color: NV}}>finds polyps by itself.</div>
      <div style={{fontFamily: MONO, fontWeight: 400, fontSize: 26, letterSpacing: 0, marginTop: 18, color: TEXT}}>NVIDIA Isaac Sim · TensorRT · digital-twin colon</div>
    </div>
  );
};

const Outro: React.FC = () => {
  const f = useCurrentFrame();
  const o = ramp(f, INTRO + MAIN, INTRO + MAIN + 0.8);
  if (o < 0.01) return null;
  const s = T.summary;
  return (
    <AbsoluteFill style={{opacity: o, background: `rgba(5,7,12,${0.95 * o})`}}>
      <div style={{position: 'absolute', left: 64, right: 64, top: 230, fontFamily: SANS, fontWeight: 800, fontSize: 96, letterSpacing: -3, color: TEXT, lineHeight: 1}}>
        {s.found_ids.length} of {s.lesions.length}<div style={{color: NV}}>polyps found.</div>
      </div>
      <div style={{position: 'absolute', left: 64, right: 64, top: 520, fontFamily: MONO, fontSize: 30, lineHeight: 1.7, color: TEXT}}>
        {EVAL.all3} of {EVAL.runs} runs: all 3 found · {s.seconds.toFixed(0)} s for 40 cm<br />
        false alarms: {EVAL.falseAlarms} of {EVAL.alarms} · polyp-free colon: {EVAL.controlFalse} in {EVAL.controlS} s
      </div>
      <div style={{position: 'absolute', left: 64, right: 64, top: 700, fontFamily: MONO, fontSize: 24, lineHeight: 1.7, color: DIM}}>
        NVIDIA Isaac Sim 6.2 (RTX) · TensorRT FP16<br />Detector: real colonoscopy + 1,500 synthetic Isaac frames<br />
        Test polyps never seen in training · kinematic scope, not soft-body
      </div>
      <div style={{position: 'absolute', left: 64, right: 64, bottom: 80, fontFamily: MONO, fontSize: 22, lineHeight: 1.6, color: DIM}}>
        Research demo · not a medical device<br />Zalkarbek Arapbaev · physician & CV engineer
      </div>
    </AbsoluteFill>
  );
};

export const Robot: React.FC = () => (
  <AbsoluteFill style={{background: BG}}>
    <Inside />
    <XRay />
    <div style={{position: 'absolute', left: 0, right: 0, top: TOP_H - 2, height: 4, background: `linear-gradient(90deg, ${NV}, ${AMBER})`}} />
    <FoundCard />
    <Intro />
    <Outro />
  </AbsoluteFill>
);
