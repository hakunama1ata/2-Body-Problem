// viewer.js
import * as THREE from 'https://unpkg.com/three@0.159.0/build/three.module.js';
import { OrbitControls } from 'https://unpkg.com/three@0.159.0/examples/jsm/controls/OrbitControls.js';

const canvas = document.getElementById('canvas');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio || 1);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, window.innerWidth/window.innerHeight, 0.1, 1000);
camera.position.set(0, 8, 30);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

// fullscreen quad shader that renders a visual black hole at origin (or at cached position)
const geom = new THREE.PlaneGeometry(2, 2);

const frag = `
precision highp float;
uniform vec2 u_res;
uniform vec3 u_bh_pos;
uniform float u_bh_mass;
uniform float u_time;

float rand(vec2 p){ return fract(sin(dot(p,vec2(12.9898,78.233)))*43758.5453); }
float star(vec2 uv){
  float s=0.0;
  for(int i=0;i<3;i++){
    vec2 p = uv*(float(i)+1.0)*6.0;
    s += smoothstep(0.998,1.0,rand(floor(p)));
  }
  return clamp(s,0.,1.);
}

void main(){
  vec2 uv = (gl_FragCoord.xy / u_res.xy) * 2.0 - 1.0;
  uv.x *= u_res.x / u_res.y;
  // simple mapping: pretend black hole is at screen center for demo
  vec2 coord = uv;
  float r = length(coord);
  float brightness = star(coord*6.0);
  // visual event horizon radius ~ f(M)
  float rs = 0.02 * u_bh_mass; // tuned scale for visualization
  vec3 color = vec3(0.01, 0.02, 0.04) + vec3(1.0)*brightness;
  if (r < rs) {
    color = vec3(0.0); // black
  } else {
    // glow
    color += vec3(1.0,0.45,0.15) * exp(-pow(r - rs, 2.0)*40.0);
  }
  gl_FragColor = vec4(color, 1.0);
}
`;

const material = new THREE.ShaderMaterial({
  vertexShader: `
    varying vec2 vUv;
    void main(){ vUv = uv; gl_Position = vec4(position, 1.0); }
  `,
  fragmentShader: frag,
  uniforms: {
    u_res: { value: new THREE.Vector2(window.innerWidth, window.innerHeight) },
    u_bh_pos: { value: new THREE.Vector3(0,0,0) },
    u_bh_mass: { value: 0.0 },
    u_time: { value: 0.0 }
  }
});
const quad = new THREE.Mesh(geom, material);
scene.add(quad);

// optional: small sphere to mark black hole world position in 3D scene (for orbiting camera)
const sphere = new THREE.Mesh(new THREE.SphereGeometry(0.5, 24, 24), new THREE.MeshBasicMaterial({color:0x000000}));
sphere.visible = false;
scene.add(sphere);

// load cached black hole if exists
async function loadCache() {
  try {
    const r = await fetch('cache/blackhole.json');
    if (!r.ok) {
      console.log('No blackhole cache found (HTTP ' + r.status + ') — showing fallback.');
      return;
    }
    const data = await r.json();
    console.log('Loaded blackhole cache:', data);
    // set shader uniform mass (scale for display)
    material.uniforms.u_bh_mass.value = Number(data.merged_mass) / 1e28; // scale down for visibility
    sphere.position.set(...(data.merged_pos.map(v => v / 1e9))); // crude scaling to world coords
    sphere.visible = true;
  } catch (err) {
    console.warn('Could not load cache:', err);
  }
}
loadCache();

// handle resize
window.addEventListener('resize', () => {
  renderer.setSize(window.innerWidth, window.innerHeight);
  material.uniforms.u_res.value.set(window.innerWidth, window.innerHeight);
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
});

let last = performance.now();
function animate(now) {
  const t = now * 0.001;
  material.uniforms.u_time.value = t;
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}
requestAnimationFrame(animate);
