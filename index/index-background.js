//
// Thanks to Jos Stam's paper "Real-Time Fluid Dynamics for Games"
// https://www.dgp.toronto.edu/public_user/stam/reality/Research/pdf/GDC03.pdf
//
// TODO: 
// Wait for page load before starting. Red black Jacobian solver. Staggered grid velocities.
//

const k_ResolutionScale = 0.125;

const k_VelocityDiffuseSteps = 20;
const k_PressureSteps = 20;

const k_ObstacleInstanceSize = 24;

let g_BackgroundRenderer;

document.addEventListener("DOMContentLoaded", () => {
	let canvas = document.getElementById("funny-background-canvas");
	g_BackgroundRenderer = new BackgroundRenderer();
	g_BackgroundRenderer.init(canvas);
});

const k_VertexShader = `
struct VertexOut {
  @builtin(position) position : vec4f,
};

const k_Positions = array<vec3f, 4>(
  vec3f( 1.0, -1.0, 0.0),
  vec3f( 1.0,  1.0, 0.0),
  vec3f(-1.0, -1.0, 0.0),
  vec3f(-1.0,  1.0, 0.0),
);

@vertex
fn vertex_main(@builtin(vertex_index) vertexIndex : u32) -> VertexOut {
  var output : VertexOut;
  let pos = k_Positions[vertexIndex];
  output.position = vec4f(pos, 1.0);
  return output;
}
`;

// Add forces from force texture to velocity texture
// Additive blending required
// Renders to v0_tex
const k_AddForcesShader = `
struct Params {
  resolution : vec2<i32>,
  dT : f32,
  _pad : i32,
};

@group(0) @binding(0) var<uniform> params : Params;
@group(1) @binding(0) var sources : texture_storage_2d<rg32float, read>;

@fragment
fn fragment_main(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec2<f32> {
  let texelCoord = vec2<i32>(fragCoord.xy);
  let source 	 = textureLoad(sources, texelCoord).xy;
  return params.dT * source; // Additive blending required
}
`;

// Jacobian solver step to diffuse velocity
// Renders to v1_tex, swap each iteration
const k_DiffuseVelocityStepShader = `
override viscosity: f32 = 0.2; // m^2/s

struct Params {
  resolution : vec2<i32>,
  dT : f32,
  _pad : i32,
};

@group(0) @binding(0) var<uniform> params : Params;
@group(1) @binding(0) var v0_tex : texture_storage_2d<rg32float, read>;

@fragment
fn fragment_main(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec2<f32> {
  let texelCoord 	= vec2<i32>(fragCoord.xy);
  var leftCoord 	= texelCoord + vec2<i32>(-1,  0);
  var rightCoord 	= texelCoord + vec2<i32>( 1,  0);
  var downCoord 	= texelCoord + vec2<i32>( 0, -1);
  var upCoord 		= texelCoord + vec2<i32>( 0,  1);

  // Wrap around
  let dim = vec2<i32>(textureDimensions(v0_tex));
  leftCoord  = (leftCoord  + dim) % dim;
  rightCoord = (rightCoord + dim) % dim;
  downCoord  = (downCoord  + dim) % dim;
  upCoord    = (upCoord    + dim) % dim;

  let a = viscosity * params.dT; // grid size is 1

  let v0 		= textureLoad(v0_tex, texelCoord).xy;
  let v0Left 	= textureLoad(v0_tex, leftCoord).xy;
  let v0Right 	= textureLoad(v0_tex, rightCoord).xy;
  let v0Down 	= textureLoad(v0_tex, downCoord).xy;
  let v0Up 		= textureLoad(v0_tex, upCoord).xy;

  let v1 = (v0 + a * (v0Left + v0Right + v0Down + v0Up)) / (1.0 + 4.0 * a);

  return v1;
}
`;

// Renders to divergence tex
const k_GetDivergenceShader = `
@group(0) @binding(0) var v_tex : texture_storage_2d<rg32float, read>;

@fragment
fn fragment_main(@builtin(position) fragCoord: vec4<f32>) -> @location(0) f32 {
  let texelCoord 	= vec2<i32>(fragCoord.xy);
  var leftCoord 	= texelCoord + vec2<i32>(-1,  0);
  var rightCoord 	= texelCoord + vec2<i32>( 1,  0);
  var downCoord 	= texelCoord + vec2<i32>( 0, -1);
  var upCoord 		= texelCoord + vec2<i32>( 0,  1);

  // Wrap around
  let dim = vec2<i32>(textureDimensions(v_tex));
  leftCoord  = (leftCoord  + dim) % dim;
  rightCoord = (rightCoord + dim) % dim;
  downCoord  = (downCoord  + dim) % dim;
  upCoord    = (upCoord    + dim) % dim;

  let vLeft 	= textureLoad(v_tex, leftCoord).xy;
  let vRight 	= textureLoad(v_tex, rightCoord).xy;
  let vDown 	= textureLoad(v_tex, downCoord).xy;
  let vUp 		= textureLoad(v_tex, upCoord).xy;

  let div = -0.5 * (vRight.x - vLeft.x + vUp.y - vDown.y);

  return div;
}
`;

// Jacobian solver step to find pressure
// Renders to p1_tex, swap each iteration
const k_CalcPressureStepShader = `
@group(0) @binding(0) var p0_tex : texture_storage_2d<r32float, read>;
@group(1) @binding(0) var div_tex : texture_storage_2d<r32float, read>;

@fragment
fn fragment_main(@builtin(position) fragCoord: vec4<f32>) -> @location(0) f32 {
  let texelCoord 	= vec2<i32>(fragCoord.xy);
  var leftCoord 	= texelCoord + vec2<i32>(-1,  0);
  var rightCoord 	= texelCoord + vec2<i32>( 1,  0);
  var downCoord 	= texelCoord + vec2<i32>( 0, -1);
  var upCoord 		= texelCoord + vec2<i32>( 0,  1);

  // Wrap around
  let dim = vec2<i32>(textureDimensions(p0_tex));
  leftCoord  = (leftCoord  + dim) % dim;
  rightCoord = (rightCoord + dim) % dim;
  downCoord  = (downCoord  + dim) % dim;
  upCoord    = (upCoord    + dim) % dim;

  let div 		= textureLoad(div_tex, texelCoord).x;
  let p0Left 	= textureLoad(p0_tex,  leftCoord).x;
  let p0Right 	= textureLoad(p0_tex,  rightCoord).x;
  let p0Down 	= textureLoad(p0_tex,  downCoord).x;
  let p0Up 		= textureLoad(p0_tex,  upCoord).x;

  let p1 = 0.25 * (div + (p0Left + p0Right + p0Down + p0Up));

  return p1;
}
`;

// Project velocity, make it mass conserving
// Renders to v_tex. Requires additive blending.
const k_ProjectVelocityShader = `
@group(0) @binding(0) var p_tex : texture_storage_2d<r32float, read>;

@fragment
fn fragment_main(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec2<f32> {
  let texelCoord 	= vec2<i32>(fragCoord.xy);
  var leftCoord 	= texelCoord + vec2<i32>(-1,  0);
  var rightCoord 	= texelCoord + vec2<i32>( 1,  0);
  var downCoord 	= texelCoord + vec2<i32>( 0, -1);
  var upCoord 		= texelCoord + vec2<i32>( 0,  1);

  // Wrap around
  let dim = vec2<i32>(textureDimensions(p_tex));
  leftCoord  = (leftCoord  + dim) % dim;
  rightCoord = (rightCoord + dim) % dim;
  downCoord  = (downCoord  + dim) % dim;
  upCoord    = (upCoord    + dim) % dim;

  let pLeft 	= textureLoad(p_tex,  leftCoord).x;
  let pRight 	= textureLoad(p_tex,  rightCoord).x;
  let pDown 	= textureLoad(p_tex,  downCoord).x;
  let pUp 		= textureLoad(p_tex,  upCoord).x;

  let sub = -0.5 * vec2<f32>(pRight - pLeft, pUp - pDown);

  return sub; // Requires additive blending
}
`;

// Self advection of velocity. Renders into v1_tex.
const k_AdvectVelocityShader = `
struct Params {
  resolution : vec2<i32>,
  dT : f32,
  _pad : i32,
};

@group(0) @binding(0) var<uniform> params : Params;
@group(1) @binding(0) var v0_tex : texture_2d<f32>;
@group(1) @binding(1) var v0_sampler : sampler;

@fragment
fn fragment_main(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec2<f32> {
  let uv = fragCoord.xy / vec2<f32>(params.resolution);
  let v0 = textureSample(v0_tex, v0_sampler, uv).xy;
  
  let back = -v0 * params.dT / vec2<f32>(params.resolution);
  let takeUV = uv - back * 500;

  let v1 = textureSample(v0_tex, v0_sampler, takeUV).xy;

  return v1;
}
`;

// Render vec2 as colour info for debugging
const k_DisplayVec2TexShader = `
struct Params {
  screenSize : vec2<i32>,
  resolution : vec2<i32>,
};

@group(0) @binding(0) var<uniform> params : Params;
@group(1) @binding(0) var v_tex : texture_storage_2d<rg32float, read>;

@fragment
fn fragment_main(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec4<f32> {
  let uv = vec2<f32>(fragCoord.xy / vec2<f32>(params.screenSize));
  let texCoord = vec2<i32>(uv * vec2<f32>(params.resolution));

  let v = textureLoad(v_tex, texCoord).xy;
  
  return vec4<f32>(abs(v), 0, 1);
}
`;

const k_DisplayVec1TexShader = `
struct Params {
  screenSize : vec2<i32>,
  resolution : vec2<i32>,
};

@group(0) @binding(0) var<uniform> params : Params;
@group(1) @binding(0) var v_tex : texture_storage_2d<r32float, read>;

@fragment
fn fragment_main(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec4<f32> {
  let uv = vec2<f32>(fragCoord.xy / vec2<f32>(params.screenSize));
  let texCoord = vec2<i32>(uv * vec2<f32>(params.resolution));

  let v = textureLoad(v_tex, texCoord).x;
  
  return vec4<f32>(0, 0, v, 1);
}
`;

const k_DisplayVec1uTexShader = `
struct Params {
  screenSize : vec2<i32>,
  resolution : vec2<i32>,
};

@group(0) @binding(0) var<uniform> params : Params;
@group(1) @binding(0) var v_tex : texture_storage_2d<r32uint, read>;

@fragment
fn fragment_main(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec4<f32> {
  let uv = vec2<f32>(fragCoord.xy / vec2<f32>(params.screenSize));
  let texCoord = vec2<i32>(uv * vec2<f32>(params.resolution));

  let v = textureLoad(v_tex, texCoord).x;
  
  return vec4<f32>(f32(v), f32(v), f32(v), 1);
}
`;

const k_DrawObstaclesMaskShader = `
struct VertexIn{
  @location(0) position : vec2<f32>,
  @location(1) dimensions : vec2<f32>,
};

struct VertexOut {
  @builtin(position) position : vec4<f32>,
};

const k_Positions = array<vec3f, 4>(
  vec3f(2.0, 0.0, 0.0),
  vec3f(2.0, 2.0, 0.0),
  vec3f(0.0, 0.0, 0.0),
  vec3f(0.0, 2.0, 0.0),
);

@vertex
fn vertex_main(@builtin(vertex_index) vertexIndex : u32, in : VertexIn) -> VertexOut {
  var output : VertexOut;

  let dim = vec2<f32>(in.dimensions);
  let offset = vec2<f32>(in.position);

  var pos = k_Positions[vertexIndex];
  pos *= vec3<f32>(dim, 0);
  pos -= vec3<f32>(1, 1, 0);
  pos += vec3<f32>(0, 2 - 2 * dim.y, 0);
  pos += 2 * vec3<f32>(offset.x, -offset.y, 0);
  output.position = vec4f(pos, 1.0);

  return output;
}

@fragment
fn fragment_main(@builtin(position) fragCoord: vec4<f32>) -> @location(0) u32 {
  return 1;
}
`;

const k_DrawObstaclesVelocityShader = `
struct VertexIn{
  @location(0) position : vec2<f32>,
  @location(1) dimensions : vec2<f32>,
  @location(2) velocity : vec2<f32>,
};

struct VertexOut {
  @builtin(position) position : vec4<f32>,
  @location(0) velocity : vec2<f32>,
};

const k_Positions = array<vec3f, 4>(
  vec3f(2.0, 0.0, 0.0),
  vec3f(2.0, 2.0, 0.0),
  vec3f(0.0, 0.0, 0.0),
  vec3f(0.0, 2.0, 0.0),
);

@vertex
fn vertex_main(@builtin(vertex_index) vertexIndex : u32, in : VertexIn) -> VertexOut {
  var output : VertexOut;

  let dim = vec2<f32>(in.dimensions);
  let offset = vec2<f32>(in.position);

  var pos = k_Positions[vertexIndex];
  pos *= vec3<f32>(dim, 0);
  pos -= vec3<f32>(1, 1, 0);
  pos += vec3<f32>(0, 2 - 2 * dim.y, 0);
  pos += 2 * vec3<f32>(offset.x, -offset.y, 0);
  output.position = vec4f(pos, 1.0);

  output.velocity = in.velocity;

  return output;
}

@fragment
fn fragment_main(in : VertexOut) -> @location(0) vec2<f32> {
  return vec2<f32>(in.velocity);
}
`;

// Set velocity in the middle of a shape to 0
const k_ModifyObstaclesVelocityShader = `
struct Params {
  resolution : vec2<i32>,
  dT : f32,
  _pad : i32,
};

@group(0) @binding(0) var<uniform> params : Params;
@group(1) @binding(0) var v0_tex : texture_storage_2d<rg32float, read>;
@group(2) @binding(0) var mask_tex : texture_storage_2d<r32uint, read>;

@fragment
fn fragment_main(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec2<f32> {
  let texelCoord 	= vec2<i32>(fragCoord.xy);
  var leftCoord 	= texelCoord + vec2<i32>(-1,  0);
  var rightCoord 	= texelCoord + vec2<i32>( 1,  0);
  var downCoord 	= texelCoord + vec2<i32>( 0, -1);
  var upCoord 		= texelCoord + vec2<i32>( 0,  1);

  // Wrap around
  let dim = vec2<i32>(textureDimensions(v0_tex));
  leftCoord  = (leftCoord  + dim) % dim;
  rightCoord = (rightCoord + dim) % dim;
  downCoord  = (downCoord  + dim) % dim;
  upCoord    = (upCoord    + dim) % dim;

  let mask 		= textureLoad(mask_tex, texelCoord).x;
  let maskLeft 	= textureLoad(mask_tex, leftCoord).x;
  let maskRight = textureLoad(mask_tex, rightCoord).x;
  let maskDown 	= textureLoad(mask_tex, downCoord).x;
  let maskUp 	= textureLoad(mask_tex, upCoord).x;

  let isCentre = mask && (maskLeft || maskRight || maskDown || maskUp);
  let mult = select(1, 0, isCentre);

  let v = textureLoad(v0_tex, texelCoord).x;

  return v * mult;
}
`;

class BoxObstacle {
	m_Timestep = 0;
	m_Position = [0, 0];
	m_LastPosition = [0, 0];
	m_Dimensions = [0, 0];

	constructor(x, y, w, h, timestep) {
		this.m_Timestep = timestep;
		this.m_Position = [x, y];
		this.m_LastPosition = [x, y];
		this.m_Dimensions = [w, h];
	}
}

class BackgroundRenderer {
	m_Canvas = null;
	m_Adapter = null;
	m_Device = null;
	m_Context = null;

	m_IsReady = false;

	m_RenderingVelocityA = true;
	m_RenderingPressureA = true;
	m_SimWidth = 100;
	m_SimHeight = 100;

	m_MouseU = 0;
	m_MouseV = 0;
	m_IsMouseDown = false;

	m_BoxObstaclesDict = {};

	async init(canvas) {
		console.log("Renderer init");
		this.m_Canvas = canvas;

		if (!this.m_Canvas) {
			throw Error("No canvas.");
		}

		if (!navigator.gpu) {
			throw Error("WebGPU not supported.");
		}

		if (!navigator.gpu.wgslLanguageFeatures.has("readonly_and_readwrite_storage_textures")) {
			throw Error("readonly_and_readwrite_storage_textures not supported.");
		}

		this.m_Adapter = await navigator.gpu.requestAdapter({
			powerPreference: "high-performance",
		});
		if (!this.m_Adapter) {
			throw Error("Couldn't request WebGPU adapter.");
		}

		if (!this.m_Adapter.features.has("float32-blendable")) {
			throw Error("float32-blendable not supported.");
		}

		if (!this.m_Adapter.features.has("float32-filterable")) {
			throw Error("float32-filterable not supported.");
		}

		this.m_Device = await this.m_Adapter.requestDevice({
			requiredFeatures: ["float32-blendable", "float32-filterable"]
		});
		if (!this.m_Device) {
			throw Error("Couldn't request WebGPU device.");
		}

		this.m_Context = this.m_Canvas.getContext("webgpu");
		if (!this.m_Context) {
			throw Error("Couldn't get webgpu context.");
		}
		this.m_Context.configure({
			device: this.m_Device,
			format: navigator.gpu.getPreferredCanvasFormat(),
			alphaMode: "premultiplied",
		});

		this.frame = this.frame.bind(this);

		this.createPipelines();

		const resizeObserver = new ResizeObserver(entries => {
			for (const entry of entries) {
				this.m_IsReady = false;

				const width = entry.contentRect.width;
				const height = entry.contentRect.height;

				// canvas.width = Math.floor(width * window.devicePixelRatio);
				// canvas.height = Math.floor(height * window.devicePixelRatio);

				canvas.width = Math.floor(width);
				canvas.height = Math.floor(height);

				console.log("Resize: " + this.m_Canvas.width + ", " + this.m_Canvas.height);
				this.createResources(canvas.width, canvas.height);
				this.createBindGroups();

				this.m_IsReady = true;
			}
		});

		window.addEventListener('mousemove', (event) => {
			const rect = canvas.getBoundingClientRect();
			const x = event.clientX - rect.left;
			const y = event.clientY - rect.top;
			const u = x / rect.width;
			const v = y / rect.height;
			this.m_MouseU = Math.min(Math.max(u, 0), 1);
			this.m_MouseV = Math.min(Math.max(v, 0), 1);
		});

		window.addEventListener('mousedown', (event) => {
			if (event.button === 0) {
				this.m_IsMouseDown = !this.m_IsMouseDown;
			}
		});

		resizeObserver.observe(this.m_Canvas);
		window.requestAnimationFrame(this.frame);
	}

	m_ParamsBindGroupLayout = null;
	m_Vec2StorageTexBindGroupLayout = null;
	m_Vec1StorageTexBindGroupLayout = null;
	m_Vec1uStorageTexBindGroupLayout = null;
	m_SampledTexBindGroupLayout = null;

	m_AddForcesPipeline = null;
	m_DiffuseVelocityPipeline = null;
	m_GetDivergencePipeline = null;
	m_CalcPressureStepPipeline = null;
	m_ProjectVelocityPipeline = null;
	m_AdvectVelocityPipeline = null;

	m_DisplayVec2TexPipeline = null;
	m_DisplayVec1TexPipeline = null;
	m_DisplayVec1uTexPipeline = null;

	m_DrawObstaclesMaskPipeline = null;

	createPipelines() {
		const vertexShaderModule = this.m_Device.createShaderModule({
			code: k_VertexShader,
		});

		this.m_ParamsBindGroupLayout = this.m_Device.createBindGroupLayout({
			entries: [
				{
					binding: 0,
					visibility: GPUShaderStage.FRAGMENT,
					buffer: {},
				}
			]
		});

		this.m_Vec2StorageTexBindGroupLayout = this.m_Device.createBindGroupLayout({
			entries: [
				{
					binding: 0,
					visibility: GPUShaderStage.FRAGMENT,
					storageTexture: {
						access: "read-only",
						format: "rg32float",
					},
				},
			]
		});

		this.m_Vec1StorageTexBindGroupLayout = this.m_Device.createBindGroupLayout({
			entries: [
				{
					binding: 0,
					visibility: GPUShaderStage.FRAGMENT,
					storageTexture: {
						access: "read-only",
						format: "r32float",
					},
				},
			]
		});

		this.m_Vec1uStorageTexBindGroupLayout = this.m_Device.createBindGroupLayout({
			entries: [
				{
					binding: 0,
					visibility: GPUShaderStage.FRAGMENT,
					storageTexture: {
						access: "read-only",
						format: "r32uint",
					},
				},
			]
		});

		this.m_SampledTexBindGroupLayout = this.m_Device.createBindGroupLayout({
			entries: [
				{
					binding: 0,
					visibility: GPUShaderStage.FRAGMENT,
					texture: {},
				},
				{
					binding: 1,
					visibility: GPUShaderStage.FRAGMENT,
					sampler: {},
				},
			]
		});

		const addForcesPipelineLayout = this.m_Device.createPipelineLayout({
			bindGroupLayouts: [this.m_ParamsBindGroupLayout, this.m_Vec2StorageTexBindGroupLayout]
		});

		const addForcesModule = this.m_Device.createShaderModule({
			code: k_AddForcesShader,
		});

		this.m_AddForcesPipeline = this.m_Device.createRenderPipeline({
			label: "Add forces pipeline",
			layout: addForcesPipelineLayout,
			fragment: {
				module: addForcesModule,
				targets: [{
					blend: {
						color: {
							srcFactor: "one",
							dstFactor: "one",
							operation: "add"
						},
						alpha: {
							srcFactor: "one",
							dstFactor: "zero",
							operation: "add"
						}
					},
					format: "rg32float"
				}],
			},
			vertex: {
				module: vertexShaderModule,
			},
			primitive: {
				topology: "triangle-strip",
				frontFace: "ccw",
				cullMode: "back",
			},
		});

		const diffuseVelocityPipelineLayout = this.m_Device.createPipelineLayout({
			bindGroupLayouts: [this.m_ParamsBindGroupLayout, this.m_Vec2StorageTexBindGroupLayout]
		});

		const diffuseVelocityModule = this.m_Device.createShaderModule({
			code: k_DiffuseVelocityStepShader,
		});

		this.m_DiffuseVelocityPipeline = this.m_Device.createRenderPipeline({
			label: "Diffuse velocity pipeline",
			layout: diffuseVelocityPipelineLayout,
			fragment: {
				module: diffuseVelocityModule,
				targets: [{
					blend: {
						color: {
							srcFactor: "one",
							dstFactor: "zero",
							operation: "add"
						},
						alpha: {
							srcFactor: "one",
							dstFactor: "zero",
							operation: "add"
						}
					},
					format: "rg32float"
				}],
			},
			vertex: {
				module: vertexShaderModule,
			},
			primitive: {
				topology: "triangle-strip",
				frontFace: "ccw",
				cullMode: "back",
			},
		});

		const getDivergencePipelineLayout = this.m_Device.createPipelineLayout({
			bindGroupLayouts: [this.m_Vec2StorageTexBindGroupLayout]
		});

		const getDivergenceModule = this.m_Device.createShaderModule({
			code: k_GetDivergenceShader,
		});

		this.m_GetDivergencePipeline = this.m_Device.createRenderPipeline({
			label: "Get divergence pipeline",
			layout: getDivergencePipelineLayout,
			fragment: {
				module: getDivergenceModule,
				targets: [{
					blend: {
						color: {
							srcFactor: "one",
							dstFactor: "zero",
							operation: "add"
						},
						alpha: {
							srcFactor: "one",
							dstFactor: "zero",
							operation: "add"
						}
					},
					format: "r32float"
				}],
			},
			vertex: {
				module: vertexShaderModule,
			},
			primitive: {
				topology: "triangle-strip",
				frontFace: "ccw",
				cullMode: "back",
			},
		});

		const calcPressureStepPipelineLayout = this.m_Device.createPipelineLayout({
			bindGroupLayouts: [this.m_Vec1StorageTexBindGroupLayout, this.m_Vec1StorageTexBindGroupLayout]
		});

		const calcPressureStepModule = this.m_Device.createShaderModule({
			code: k_CalcPressureStepShader,
		});

		this.m_CalcPressureStepPipeline = this.m_Device.createRenderPipeline({
			label: "Calc pressure pipeline",
			layout: calcPressureStepPipelineLayout,
			fragment: {
				module: calcPressureStepModule,
				targets: [{
					blend: {
						color: {
							srcFactor: "one",
							dstFactor: "zero",
							operation: "add"
						},
						alpha: {
							srcFactor: "one",
							dstFactor: "zero",
							operation: "add"
						}
					},
					format: "r32float"
				}],
			},
			vertex: {
				module: vertexShaderModule,
			},
			primitive: {
				topology: "triangle-strip",
				frontFace: "ccw",
				cullMode: "back",
			},
		});

		const projectVelocityPipelineLayout = this.m_Device.createPipelineLayout({
			bindGroupLayouts: [this.m_Vec1StorageTexBindGroupLayout]
		});

		const projectVelocityStepModule = this.m_Device.createShaderModule({
			code: k_ProjectVelocityShader,
		});

		this.m_ProjectVelocityPipeline = this.m_Device.createRenderPipeline({
			label: "Project velocity pipeline",
			layout: projectVelocityPipelineLayout,
			fragment: {
				module: projectVelocityStepModule,
				targets: [{
					blend: {
						color: {
							srcFactor: "one",
							dstFactor: "one",
							operation: "add"
						},
						alpha: {
							srcFactor: "one",
							dstFactor: "zero",
							operation: "add"
						}
					},
					format: "rg32float"
				}],
			},
			vertex: {
				module: vertexShaderModule,
			},
			primitive: {
				topology: "triangle-strip",
				frontFace: "ccw",
				cullMode: "back",
			},
		});

		const advectVelocityPipelineLayout = this.m_Device.createPipelineLayout({
			bindGroupLayouts: [this.m_ParamsBindGroupLayout, this.m_SampledTexBindGroupLayout]
		});

		const advectVelocityStepModule = this.m_Device.createShaderModule({
			code: k_AdvectVelocityShader,
		});

		this.m_AdvectVelocityPipeline = this.m_Device.createRenderPipeline({
			label: "Advect velocity pipeline",
			layout: advectVelocityPipelineLayout,
			fragment: {
				module: advectVelocityStepModule,
				targets: [{
					blend: {
						color: {
							srcFactor: "one",
							dstFactor: "zero",
							operation: "add"
						},
						alpha: {
							srcFactor: "one",
							dstFactor: "zero",
							operation: "add"
						}
					},
					format: "rg32float"
				}],
			},
			vertex: {
				module: vertexShaderModule,
			},
			primitive: {
				topology: "triangle-strip",
				frontFace: "ccw",
				cullMode: "back",
			},
		});

		const displayVec2TexPipelineLayout = this.m_Device.createPipelineLayout({
			bindGroupLayouts: [this.m_ParamsBindGroupLayout, this.m_Vec2StorageTexBindGroupLayout]
		});

		const displayVec2TexModule = this.m_Device.createShaderModule({
			code: k_DisplayVec2TexShader,
		});

		this.m_DisplayVec2TexPipeline = this.m_Device.createRenderPipeline({
			layout: displayVec2TexPipelineLayout,
			fragment: {
				module: displayVec2TexModule,
				targets: [{
					blend: {
						color: {
							srcFactor: "one",
							dstFactor: "zero",
							operation: "add"
						},
						alpha: {
							srcFactor: "one",
							dstFactor: "zero",
							operation: "add"
						}
					},
					format: navigator.gpu.getPreferredCanvasFormat()
				}],
			},
			vertex: {
				module: vertexShaderModule,
			},
			primitive: {
				topology: "triangle-strip",
				frontFace: "ccw",
				cullMode: "back",
			},
		});

		const displayVec1TexPipelineLayout = this.m_Device.createPipelineLayout({
			bindGroupLayouts: [this.m_ParamsBindGroupLayout, this.m_Vec1StorageTexBindGroupLayout]
		});

		const displayVec1TexModule = this.m_Device.createShaderModule({
			code: k_DisplayVec1TexShader,
		});

		this.m_DisplayVec1TexPipeline = this.m_Device.createRenderPipeline({
			layout: displayVec1TexPipelineLayout,
			fragment: {
				module: displayVec1TexModule,
				targets: [{
					blend: {
						color: {
							srcFactor: "one",
							dstFactor: "zero",
							operation: "add"
						},
						alpha: {
							srcFactor: "one",
							dstFactor: "zero",
							operation: "add"
						}
					},
					format: navigator.gpu.getPreferredCanvasFormat()
				}],
			},
			vertex: {
				module: vertexShaderModule,
			},
			primitive: {
				topology: "triangle-strip",
				frontFace: "ccw",
				cullMode: "back",
			},
		});

		const displayVec1uTexPipelineLayout = this.m_Device.createPipelineLayout({
			bindGroupLayouts: [this.m_ParamsBindGroupLayout, this.m_Vec1uStorageTexBindGroupLayout]
		});

		const displayVec1uTexModule = this.m_Device.createShaderModule({
			code: k_DisplayVec1uTexShader,
		});

		this.m_DisplayVec1uTexPipeline = this.m_Device.createRenderPipeline({
			layout: displayVec1uTexPipelineLayout,
			fragment: {
				module: displayVec1uTexModule,
				targets: [{
					blend: {
						color: {
							srcFactor: "one",
							dstFactor: "zero",
							operation: "add"
						},
						alpha: {
							srcFactor: "one",
							dstFactor: "zero",
							operation: "add"
						}
					},
					format: navigator.gpu.getPreferredCanvasFormat()
				}],
			},
			vertex: {
				module: vertexShaderModule,
			},
			primitive: {
				topology: "triangle-strip",
				frontFace: "ccw",
				cullMode: "back",
			},
		});

		const drawObstaclesMaskPipelineLayout = this.m_Device.createPipelineLayout({
			bindGroupLayouts: []
		});

		const drawObstaclesMaskModule = this.m_Device.createShaderModule({
			code: k_DrawObstaclesMaskShader,
		});

		this.m_DrawObstaclesMaskPipeline = this.m_Device.createRenderPipeline({
			label: "Draw obstacles mask pipeline",
			layout: drawObstaclesMaskPipelineLayout,
			fragment: {
				module: drawObstaclesMaskModule,
				targets: [{
					format: "r32uint"
				}],
			},
			vertex: {
				module: drawObstaclesMaskModule,
				buffers: [
					{
						stepMode: "instance",
						arrayStride: 24,
						attributes: [
							{
								shaderLocation: 0,
								offset: 0,
								format: "float32x2",
							},
							{
								shaderLocation: 1,
								offset: 8,
								format: "float32x2",
							},
						],
					}
				]
			},
			primitive: {
				topology: "triangle-strip",
				frontFace: "ccw",
				cullMode: "back",
			},
		});

		const drawObstaclesVelocityPipelineLayout = this.m_Device.createPipelineLayout({
			bindGroupLayouts: []
		});

		const drawObstaclesVelocityModule = this.m_Device.createShaderModule({
			code: k_DrawObstaclesVelocityShader,
		});

		this.m_DrawObstaclesVelocityPipeline = this.m_Device.createRenderPipeline({
			label: "Draw obstacles velocity pipeline",
			layout: drawObstaclesVelocityPipelineLayout,
			fragment: {
				module: drawObstaclesVelocityModule,
				targets: [{
					format: "rg32float"
				}],
			},
			vertex: {
				module: drawObstaclesVelocityModule,
				buffers: [
					{
						stepMode: "instance",
						arrayStride: 24,
						attributes: [
							{
								shaderLocation: 0,
								offset: 0,
								format: "float32x2",
							},
							{
								shaderLocation: 1,
								offset: 8,
								format: "float32x2",
							},
							{
								shaderLocation: 2,
								offset: 16,
								format: "float32x2",
							},
						],
					}
				]
			},
			primitive: {
				topology: "triangle-strip",
				frontFace: "ccw",
				cullMode: "back",
			},
		});
	}

	m_ParamsBuffer = null;
	m_DebugParamsBuffer = null;
	m_ForcesTex = null;
	m_ForcesTexView = null;
	m_VelocityTexA = null;
	m_VelocityTexViewA = null;
	m_VelocityTexB = null;
	m_VelocityTexViewB = null;
	m_DivergenceTex = null;
	m_DivergenceTexView = null;
	m_PressureTexA = null;
	m_PressureTexViewA = null;
	m_PressureTexB = null;
	m_PressureTexViewB = null;
	m_Sampler = null;

	m_ObstacleMask = null;
	m_ObstacleMaskView = null;
	m_ObstacleVelocity = null;
	m_ObstacleVelocityView = null;
	m_ObstacleInstanceBuffer = null;

	createResources(width, height) {
		this.m_SimWidth = width * k_ResolutionScale;
		this.m_SimHeight = height * k_ResolutionScale;

		this.m_ParamsBuffer = this.m_Device.createBuffer({
			size: 16,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
			label: "Params Buffer"
		});

		this.m_DebugParamsBuffer = this.m_Device.createBuffer({
			size: 16,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
			label: "Debug Params Buffer"
		});

		this.m_ForcesTex = this.m_Device.createTexture({
			format: "rg32float",
			size: [this.m_SimWidth, this.m_SimHeight],
			usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST,
			label: "Forces Texture"
		});
		this.m_ForcesTexView = this.m_ForcesTex.createView();

		this.m_VelocityTexA = this.m_Device.createTexture({
			format: "rg32float",
			size: [this.m_SimWidth, this.m_SimHeight],
			usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
			label: "Velocity Texture A",

		});
		this.m_VelocityTexViewA = this.m_VelocityTexA.createView();

		this.m_VelocityTexB = this.m_Device.createTexture({
			format: "rg32float",
			size: [this.m_SimWidth, this.m_SimHeight],
			usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
			label: "Velocity Texture B"
		});
		this.m_VelocityTexViewB = this.m_VelocityTexB.createView();

		this.m_DivergenceTex = this.m_Device.createTexture({
			format: "r32float",
			size: [this.m_SimWidth, this.m_SimHeight],
			usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING,
			label: "Divergence Texture"
		});
		this.m_DivergenceTexView = this.m_DivergenceTex.createView();

		this.m_PressureTexA = this.m_Device.createTexture({
			format: "r32float",
			size: [this.m_SimWidth, this.m_SimHeight],
			usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING,
			label: "Pressure Texture A"
		});
		this.m_PressureTexViewA = this.m_PressureTexA.createView();

		this.m_PressureTexB = this.m_Device.createTexture({
			format: "r32float",
			size: [this.m_SimWidth, this.m_SimHeight],
			usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING,
			label: "Pressure Texture B"
		});
		this.m_PressureTexViewB = this.m_PressureTexB.createView();

		this.m_Sampler = this.m_Device.createSampler({
			addressModeU: "repeat",
			addressModeV: "repeat",
			magFilter: "linear",
			minFilter: "linear",
		});

		this.m_ObstacleMask = this.m_Device.createTexture({
			format: "r32uint",
			size: [this.m_SimWidth, this.m_SimHeight],
			usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING,
			label: "Obstacle Mask"
		});
		this.m_ObstacleMaskView = this.m_ObstacleMask.createView();

		this.m_ObstacleVelocity = this.m_Device.createTexture({
			format: "rg32float",
			size: [this.m_SimWidth, this.m_SimHeight],
			usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING,
			label: "Obstacle Velocity"
		});
		this.m_ObstacleVelocityView = this.m_ObstacleVelocity.createView();

		this.createObstacleInstanceBuffer();
	}

	createObstacleInstanceBuffer(objCount = 32) {
		this.m_ObstacleInstanceBuffer = this.m_Device.createBuffer({
			label: "Object instance buffer",
			size: k_ObstacleInstanceSize * objCount,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.VERTEX
		});
	}

	m_ParamsBindGroup = null;
	m_DebugParamsBindGroup = null;
	m_ForcesStorageTexBindGroup = null;
	m_VelocityStorageTexBindGroupA = null;
	m_VelocityStorageTexBindGroupB = null;
	m_VelocitySampledTexBindGroupA = null;
	m_VelocitySampledTexBindGroupB = null;
	m_DivergenceTexBindGroup = null;
	m_PressureTexBindGroupA = null;
	m_PressureTexBindGroupB = null;

	m_ObstacleMaskBindGroup = null;
	m_ObstacleVelocityBindGroup = null;

	createBindGroups() {
		this.m_ParamsBindGroup = this.m_Device.createBindGroup({
			layout: this.m_ParamsBindGroupLayout,
			entries: [
				{
					binding: 0,
					resource: this.m_ParamsBuffer
				}
			]
		});

		this.m_DebugParamsBindGroup = this.m_Device.createBindGroup({
			layout: this.m_ParamsBindGroupLayout,
			entries: [
				{
					binding: 0,
					resource: this.m_DebugParamsBuffer
				}
			]
		});

		this.m_ForcesStorageTexBindGroup = this.m_Device.createBindGroup({
			layout: this.m_Vec2StorageTexBindGroupLayout,
			entries: [
				{
					binding: 0,
					resource: this.m_ForcesTex
				}
			]
		});

		this.m_VelocityStorageTexBindGroupA = this.m_Device.createBindGroup({
			layout: this.m_Vec2StorageTexBindGroupLayout,
			entries: [
				{
					binding: 0,
					resource: this.m_VelocityTexA
				}
			]
		});

		this.m_VelocityStorageTexBindGroupB = this.m_Device.createBindGroup({
			layout: this.m_Vec2StorageTexBindGroupLayout,
			entries: [
				{
					binding: 0,
					resource: this.m_VelocityTexB
				}
			]
		});

		this.m_VelocitySampledTexBindGroupA = this.m_Device.createBindGroup({
			layout: this.m_SampledTexBindGroupLayout,
			entries: [
				{
					binding: 0,
					resource: this.m_VelocityTexViewA
				},
				{
					binding: 1,
					resource: this.m_Sampler
				}
			]
		});

		this.m_VelocitySampledTexBindGroupB = this.m_Device.createBindGroup({
			layout: this.m_SampledTexBindGroupLayout,
			entries: [
				{
					binding: 0,
					resource: this.m_VelocityTexViewB
				},
				{
					binding: 1,
					resource: this.m_Sampler
				}
			]
		});

		this.m_DivergenceTexBindGroup = this.m_Device.createBindGroup({
			layout: this.m_Vec1StorageTexBindGroupLayout,
			entries: [
				{
					binding: 0,
					resource: this.m_DivergenceTex
				}
			]
		});

		this.m_PressureTexBindGroupA = this.m_Device.createBindGroup({
			layout: this.m_Vec1StorageTexBindGroupLayout,
			entries: [
				{
					binding: 0,
					resource: this.m_PressureTexA
				}
			]
		});

		this.m_PressureTexBindGroupB = this.m_Device.createBindGroup({
			layout: this.m_Vec1StorageTexBindGroupLayout,
			entries: [
				{
					binding: 0,
					resource: this.m_PressureTexB
				}
			]
		});

		this.m_ObstacleMaskBindGroup = this.m_Device.createBindGroup({
			layout: this.m_Vec1uStorageTexBindGroupLayout,
			entries: [
				{
					binding: 0,
					resource: this.m_ObstacleMask
				}
			]
		});

		this.m_ObstacleVelocityBindGroup = this.m_Device.createBindGroup({
			layout: this.m_Vec2StorageTexBindGroupLayout,
			entries: [
				{
					binding: 0,
					resource: this.m_ObstacleVelocity
				}
			]
		});
	}

	async frame(timestep) {
		if (!this.m_IsReady) {
			this.m_PrevTimeStep = timestep;
			window.requestAnimationFrame(this.frame);
			return;
		}

		this.updateObstacles();

		let deltaT = (timestep - this.m_PrevTimeStep) / 1000;
		this.m_PrevTimeStep = timestep;

		// Setup frame
		const renderTexture = this.m_Context.getCurrentTexture();
		const renderView = renderTexture.createView();

		const commandEncoder = this.m_Device.createCommandEncoder();

		this.drawObstacles(commandEncoder, deltaT);

		// Set params
		{
			let paramsView = new DataView(new ArrayBuffer(16));

			paramsView.setInt32(0, this.m_SimWidth, true);
			paramsView.setInt32(4, this.m_SimHeight, true);
			paramsView.setFloat32(8, deltaT, true);
			paramsView.setInt32(12, 0, true);

			this.m_Device.queue.writeBuffer(this.m_ParamsBuffer, 0, paramsView);
		}

		// Clear forces
		{
			const clearForcesCommandEncoder = this.m_Device.createCommandEncoder();
			const clearForcesPass = clearForcesCommandEncoder.beginRenderPass({
				colorAttachments: [{
					loadOp: "clear",
					storeOp: "store",
					view: this.m_ForcesTexView,
					clearValue: [0, 0, 0, 0],
				}],
			});

			clearForcesPass.end();
			this.m_Device.queue.submit([clearForcesCommandEncoder.finish()]);
		}

		// Set forces at mouse
		if (this.m_IsMouseDown) {
			const scale = 200;
			let forcesView = new DataView(new ArrayBuffer(16));
			forcesView.setFloat32(0, 0, true);
			forcesView.setFloat32(4, scale, true);

			forcesView.setFloat32(8, 0, true);
			forcesView.setFloat32(12, 0, true);

			const x = Math.min(Math.max(this.m_MouseU * this.m_SimWidth, 0), this.m_SimWidth - 1);
			const y = Math.min(Math.max(this.m_MouseV * this.m_SimHeight, 0), this.m_SimHeight - 2);

			this.m_Device.queue.writeTexture(
				{
					origin: [x, y, 0],
					texture: this.m_ForcesTex
				},
				forcesView,
				{
					bytesPerRow: 8,
				},
				{
					width: 1,
					height: 2,
				}
			);
		}

		// Set debug params
		{
			let paramsView = new DataView(new ArrayBuffer(16));

			paramsView.setInt32(0, renderTexture.width, true);
			paramsView.setInt32(4, renderTexture.height, true);
			paramsView.setInt32(8, this.m_SimWidth, true);
			paramsView.setInt32(12, this.m_SimHeight, true);

			this.m_Device.queue.writeBuffer(this.m_DebugParamsBuffer, 0, paramsView);
		}

		// Add forces
		{
			const addForcesPass = commandEncoder.beginRenderPass({
				colorAttachments: [{
					loadOp: "load",
					storeOp: "store",
					view: this.m_RenderingVelocityA ? this.m_VelocityTexViewA : this.m_VelocityTexViewB,
				}],
			});

			addForcesPass.setViewport(0, 0, this.m_SimWidth, this.m_SimHeight, 0, 1);
			addForcesPass.setScissorRect(0, 0, this.m_SimWidth, this.m_SimHeight);
			addForcesPass.setPipeline(this.m_AddForcesPipeline);

			addForcesPass.setBindGroup(0, this.m_ParamsBindGroup);
			addForcesPass.setBindGroup(1, this.m_ForcesStorageTexBindGroup);

			addForcesPass.draw(4);
			addForcesPass.end();
		}

		// Diffuse velocity
		for (let i = 0; i < k_VelocityDiffuseSteps; ++i) {
			this.m_RenderingVelocityA = !this.m_RenderingVelocityA;

			const diffuseStepPass = commandEncoder.beginRenderPass({
				colorAttachments: [{
					loadOp: "load",
					storeOp: "store",
					view: this.m_RenderingVelocityA ? this.m_VelocityTexViewA : this.m_VelocityTexViewB,
				}],
			});

			diffuseStepPass.setViewport(0, 0, this.m_SimWidth, this.m_SimHeight, 0, 1);
			diffuseStepPass.setScissorRect(0, 0, this.m_SimWidth, this.m_SimHeight);
			diffuseStepPass.setPipeline(this.m_DiffuseVelocityPipeline);

			diffuseStepPass.setBindGroup(0, this.m_ParamsBindGroup);
			diffuseStepPass.setBindGroup(1, this.m_RenderingVelocityA ? this.m_VelocityStorageTexBindGroupB : this.m_VelocityStorageTexBindGroupA);

			diffuseStepPass.draw(4);
			diffuseStepPass.end();
		}

		this.projectStep(commandEncoder);

		// Advect velocity
		{
			this.m_RenderingVelocityA = !this.m_RenderingVelocityA;

			const advectVelocityPass = commandEncoder.beginRenderPass({
				colorAttachments: [{
					loadOp: "load",
					storeOp: "store",
					view: this.m_RenderingVelocityA ? this.m_VelocityTexViewA : this.m_VelocityTexViewB,
				}],
			});

			advectVelocityPass.setViewport(0, 0, this.m_SimWidth, this.m_SimHeight, 0, 1);
			advectVelocityPass.setScissorRect(0, 0, this.m_SimWidth, this.m_SimHeight);
			advectVelocityPass.setPipeline(this.m_AdvectVelocityPipeline);

			advectVelocityPass.setBindGroup(0, this.m_ParamsBindGroup);
			advectVelocityPass.setBindGroup(1, this.m_RenderingVelocityA ? this.m_VelocitySampledTexBindGroupB : this.m_VelocitySampledTexBindGroupA);

			advectVelocityPass.draw(4);
			advectVelocityPass.end();
		}

		this.projectStep(commandEncoder);

		// Display obstacle velocity
		{
			const displayObstaclesPass = commandEncoder.beginRenderPass({
				colorAttachments: [{
					loadOp: "clear",
					storeOp: "store",
					view: renderView,
					clearValue: [0, 0, 0, 1],
				}],
			});

			displayObstaclesPass.setViewport(0, 0, renderTexture.width, renderTexture.height, 0, 1);
			displayObstaclesPass.setScissorRect(0, 0, renderTexture.width, renderTexture.height);
			displayObstaclesPass.setPipeline(this.m_DisplayVec2TexPipeline);

			displayObstaclesPass.setBindGroup(0, this.m_DebugParamsBindGroup);
			displayObstaclesPass.setBindGroup(1, this.m_ObstacleVelocityBindGroup);

			displayObstaclesPass.draw(4);
			displayObstaclesPass.end();
		}

		this.m_Device.queue.submit([commandEncoder.finish()]);

		window.requestAnimationFrame(this.frame);
	}

	projectStep(commandEncoder) {
		// Get divergence
		{
			const divergencePass = commandEncoder.beginRenderPass({
				colorAttachments: [{
					loadOp: "load",
					storeOp: "store",
					view: this.m_DivergenceTexView,
				}],
			});

			divergencePass.setViewport(0, 0, this.m_SimWidth, this.m_SimHeight, 0, 1);
			divergencePass.setScissorRect(0, 0, this.m_SimWidth, this.m_SimHeight);
			divergencePass.setPipeline(this.m_GetDivergencePipeline);

			divergencePass.setBindGroup(0, this.m_RenderingVelocityA ? this.m_VelocityStorageTexBindGroupA : this.m_VelocityStorageTexBindGroupB);

			divergencePass.draw(4);
			divergencePass.end();
		}

		// Clear pressure
		// const pressureClearPass = commandEncoder.beginRenderPass({
		// 	colorAttachments: [{
		// 		loadOp: "clear",
		// 		storeOp: "store",
		// 		view: this.m_RenderingPressureA ? this.m_PressureTexViewB : this.m_PressureTexViewA,
		// 		clearValue: [0, 0, 0, 1]
		// 	}],
		// });
		// pressureClearPass.end();

		// Get pressure
		for (let i = 0; i < k_PressureSteps; ++i) {
			this.m_RenderingPressureA = !this.m_RenderingPressureA;

			const pressureStepPass = commandEncoder.beginRenderPass({
				colorAttachments: [{
					loadOp: "load",
					storeOp: "store",
					view: this.m_RenderingPressureA ? this.m_PressureTexViewA : this.m_PressureTexViewB,
				}],
			});

			pressureStepPass.setViewport(0, 0, this.m_SimWidth, this.m_SimHeight, 0, 1);
			pressureStepPass.setScissorRect(0, 0, this.m_SimWidth, this.m_SimHeight);
			pressureStepPass.setPipeline(this.m_CalcPressureStepPipeline);

			pressureStepPass.setBindGroup(0, this.m_RenderingPressureA ? this.m_PressureTexBindGroupB : this.m_PressureTexBindGroupA);
			pressureStepPass.setBindGroup(1, this.m_DivergenceTexBindGroup);

			pressureStepPass.draw(4);
			pressureStepPass.end();
		}

		// Project velocity
		{
			const projectPass = commandEncoder.beginRenderPass({
				colorAttachments: [{
					loadOp: "load",
					storeOp: "store",
					view: this.m_RenderingVelocityA ? this.m_VelocityTexViewA : this.m_VelocityTexViewB,
				}],
			});

			projectPass.setViewport(0, 0, this.m_SimWidth, this.m_SimHeight, 0, 1);
			projectPass.setScissorRect(0, 0, this.m_SimWidth, this.m_SimHeight);
			projectPass.setPipeline(this.m_ProjectVelocityPipeline);

			projectPass.setBindGroup(0, this.m_RenderingPressureA ? this.m_PressureTexBindGroupA : this.m_PressureTexBindGroupB);

			projectPass.draw(4);
			projectPass.end();
		}
	}

	updateObstacles(timestep) {
		const elements = document.querySelectorAll("fluid-box");
		elements.forEach(el => {
			const rect = el.getBoundingClientRect();
			if (this.m_BoxObstaclesDict[el.id]) {
				this.m_BoxObstaclesDict[el.id].m_LastPosition = [this.m_BoxObstaclesDict[el.id].m_Position[0], this.m_BoxObstaclesDict[el.id].m_Position[1]];
				this.m_BoxObstaclesDict[el.id].m_Position = [rect.left / this.m_Canvas.width, rect.top / this.m_Canvas.height];
				this.m_BoxObstaclesDict[el.id].m_Dimensions = [rect.width / this.m_Canvas.width, rect.height / this.m_Canvas.height];

				this.m_Timestep = timestep;
			} else {
				console.log(`${rect.left}, ${rect.top}, ${rect.width}, ${rect.height}`);
				this.m_BoxObstaclesDict[el.id] = new BoxObstacle(rect.left, rect.top, rect.width, rect.height, timestep);
			}
		});

		Object.entries(this.m_BoxObstaclesDict).forEach(([key, value]) => {
			if (value.m_Timestep < timestep) {
				console.log("delete");
				delete this.m_BoxObstaclesDict[key];
			}
		});
	}

	drawObstacles(commandEncoder, deltaT) {
		const obstacles = Object.entries(this.m_BoxObstaclesDict);

		const capacity = this.m_ObstacleInstanceBuffer.size / k_ObstacleInstanceSize;
		if (capacity < obstacles.length) {
			this.createObstacleInstanceBuffer(capacity * 2);
		}

		let instanceData = new DataView(new ArrayBuffer(obstacles.length * k_ObstacleInstanceSize));

		let offset = 0;
		obstacles.forEach(([key, value]) => {
			instanceData.setFloat32(offset + 0, value.m_Position[0], true);
			instanceData.setFloat32(offset + 4, value.m_Position[1], true);
			instanceData.setFloat32(offset + 8, value.m_Dimensions[0], true);
			instanceData.setFloat32(offset + 12, value.m_Dimensions[1], true);
			instanceData.setFloat32(offset + 16, (value.m_Position[0] - value.m_LastPosition[0]) / deltaT, true);
			instanceData.setFloat32(offset + 20, (value.m_Position[1] - value.m_LastPosition[1]) / deltaT, true);

			offset += k_ObstacleInstanceSize;
		});

		this.m_Device.queue.writeBuffer(this.m_ObstacleInstanceBuffer, 0, instanceData);

		// Draw to mask
		const maskPass = commandEncoder.beginRenderPass({
			colorAttachments: [{
				loadOp: "clear",
				storeOp: "store",
				view: this.m_ObstacleMaskView,
				clearValue: [0, 0, 0, 1]
			}],
		});

		maskPass.setViewport(0, 0, this.m_SimWidth, this.m_SimHeight, 0, 1);
		maskPass.setScissorRect(0, 0, this.m_SimWidth, this.m_SimHeight);
		maskPass.setPipeline(this.m_DrawObstaclesMaskPipeline);
		maskPass.setVertexBuffer(0, this.m_ObstacleInstanceBuffer);

		maskPass.draw(4, obstacles.length);

		maskPass.end();

		// Draw to velocity
		const velPass = commandEncoder.beginRenderPass({
			colorAttachments: [{
				loadOp: "clear",
				storeOp: "store",
				view: this.m_ObstacleVelocityView,
				clearValue: [0, 0, 0, 1]
			}],
		});

		velPass.setViewport(0, 0, this.m_SimWidth, this.m_SimHeight, 0, 1);
		velPass.setScissorRect(0, 0, this.m_SimWidth, this.m_SimHeight);
		velPass.setPipeline(this.m_DrawObstaclesVelocityPipeline);
		velPass.setVertexBuffer(0, this.m_ObstacleInstanceBuffer);

		velPass.draw(4, obstacles.length);

		velPass.end();
	}
}