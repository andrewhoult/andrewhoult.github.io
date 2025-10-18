//
// Thanks to Jos Stam's paper "Real-Time Fluid Dynamics for Games"
// https://www.dgp.toronto.edu/public_user/stam/reality/Research/pdf/GDC03.pdf
//
// TODO:
// Adopt all to texture pool.
// Compute shaders.
// Red black Jacobian solver.
// Staggered grid velocities.
//

const k_ResolutionScale = 0.5;
const k_MaxPixels = 512 * 512;

const k_VelocityDiffuseSteps = 20;
const k_PressureSteps = 20;

const k_ObstacleInstanceSize = 24;

const k_TeleportThreshold = 10;

const k_Inset = 0;

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

// Jacobian solver step to diffuse velocity
// Renders to v1_tex, swap each iteration
const k_DiffuseVelocityStepShader = `
override viscosity: f32 = 0.002; // m^2/s
override omega: f32 = 1.9;

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

  var v1 = (v0 + a * (v0Left + v0Right + v0Down + v0Up)) / (1.0 + 4.0 * a);

  // Over-relaxation
  v1 = mix(v0, v1, omega);

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
  
  return vec4<f32>(saturate(abs(v)), 0, 1);
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
  
  return vec4<f32>(0, 0, abs(v), 1);
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

  let v0		= textureLoad(v0_tex, texelCoord).xy;
  let v0Left	= textureLoad(v0_tex, leftCoord).xy;
  let v0Right	= textureLoad(v0_tex, rightCoord).xy;
  let v0Down	= textureLoad(v0_tex, downCoord).xy;
  let v0Up		= textureLoad(v0_tex, upCoord).xy;

  let mask 		= v0.x != 9999 		|| v0.y != 9999;
  let maskLeft 	= v0Left.x != 9999 	|| v0Left.y != 9999;
  let maskRight = v0Right.x != 9999 || v0Right.y != 9999;
  let maskDown 	= v0Down.x != 9999 	|| v0Down.y != 9999;
  let maskUp 	= v0Up.x != 9999 	|| v0Up.y != 9999;

  let isEdge = mask && (maskLeft != maskRight || maskDown != maskUp);

  return select(vec2(0.0, 0.0), v0, isEdge || !mask);
}
`;

// Enforce boundary condition
const k_EnforceBoundaryVelocityShader = `
struct Params {
  resolution : vec2<i32>,
  dT : f32,
  _pad : i32,
};

@group(0) @binding(0) var<uniform> params : Params;
@group(1) @binding(0) var v0_tex : texture_storage_2d<rg32float, read>;
@group(2) @binding(0) var boundvel_tex : texture_storage_2d<rg32float, read>;

@fragment
fn fragment_main(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec2<f32> {
  let texelCoord = vec2<i32>(fragCoord.xy);
  let ogVel = textureLoad(v0_tex, texelCoord).xy;
  let boundVel = textureLoad(boundvel_tex, texelCoord).xy;
  let isBound: bool = boundVel.x != 9999 || boundVel.y != 9999;

  return select(ogVel, boundVel, isBound);
}
`;

class TexRef {
	m_Format = null;
	m_Tex = null;
	m_View = null;

	m_StorageBindGroup = null;
	m_SampledBindGroup = null;

	constructor(format) {
		this.m_Format = format;
	}
}

class TexturePool {
	m_Device = null;

	m_SimWidth = 0;
	m_SimHeight = 0;
	m_Balance = 0;
	m_Dict = {};

	m_Vec2StorageTexBindGroupLayout = null;
	m_Vec1StorageTexBindGroupLayout = null;
	m_Vec1uStorageTexBindGroupLayout = null;
	m_SampledTexBindGroupLayout = null;

	constructor(device, width, height) {
		this.m_Device = device;

		this.setSize(width, height);

		// create bind group layouts
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

		// create sampler
		this.m_Sampler = this.m_Device.createSampler({
			addressModeU: "repeat",
			addressModeV: "repeat",
			magFilter: "linear",
			minFilter: "linear",
		});
	}

	setSize(width, height) {
		console.log("Resetting texture pool");
		this.m_Dict = {};
		this.m_Balance = 0;

		this.m_SimWidth = width;
		this.m_SimHeight = height;
	}

	acquire(format) {
		if (!this.m_Dict[format]) {
			this.m_Dict[format] = {
				pool: [],
			}
		}

		let texRef = null;
		if (this.m_Dict[format].pool.length == 0) {
			console.log("Creating tex: " + format);

			texRef = new TexRef(format);

			texRef.m_Tex = this.m_Device.createTexture({
				format: format,
				size: [this.m_SimWidth, this.m_SimHeight],
				usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,

			});
			texRef.m_View = texRef.m_Tex.createView();

			let layout = null;
			switch (format) {
				case "r32float":
					layout = this.m_Vec1StorageTexBindGroupLayout
					break;
				case "rg32float":
					layout = this.m_Vec2StorageTexBindGroupLayout
					break;
				case "r32uint":
					layout = this.m_Vec1uStorageTexBindGroupLayout
					break;
			}

			if (!layout) throw new Error(`Unknown format ${format}`);

			texRef.m_StorageBindGroup = this.m_Device.createBindGroup({
				layout: layout,
				entries: [
					{
						binding: 0,
						resource: texRef.m_View
					},
				]
			});

			texRef.m_SampledBindGroup = this.m_Device.createBindGroup({
				layout: this.m_SampledTexBindGroupLayout,
				entries: [
					{
						binding: 0,
						resource: texRef.m_View
					},
					{
						binding: 1,
						resource: this.m_Sampler
					}
				]
			});
		} else {
			texRef = this.m_Dict[format].pool.pop();
		}

		++this.m_Balance;
		return texRef;
	}

	release(texRef) {
		this.m_Dict[texRef.m_Format].pool.push(texRef);
		--this.m_Balance;
	}

	checkBalance() {
		if (this.m_Balance != 0) {
			console.log(`TexPool balance: ${this.m_Balance}`);
		}
		this.m_Balance = 0;
	}
}

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

	m_SimWidth = 100;
	m_SimHeight = 100;

	m_TexturePool = null;
	m_BoxObstaclesDict = {};

	m_CurrentVelocityTex = null;
	m_CurrentPressureTex = null;

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

		this.m_TexturePool = new TexturePool(this.m_Device);

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

		resizeObserver.observe(this.m_Canvas);
		window.requestAnimationFrame(this.frame);
	}

	m_ParamsBindGroupLayout = null;

	m_DiffuseVelocityPipeline = null;
	m_GetDivergencePipeline = null;
	m_CalcPressureStepPipeline = null;
	m_ProjectVelocityPipeline = null;
	m_AdvectVelocityPipeline = null;

	m_DisplayVec2TexPipeline = null;
	m_DisplayVec1TexPipeline = null;
	m_DisplayVec1uTexPipeline = null;

	m_DrawObstaclesVelocityPipeline = null;
	m_ModifyObstaclesVelocityPipeline = null;
	m_EnforceBoundaryVelocityPipeline = null;

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

		const diffuseVelocityPipelineLayout = this.m_Device.createPipelineLayout({
			label: "diffuseVelocityPipelineLayout",
			bindGroupLayouts: [this.m_ParamsBindGroupLayout, this.m_TexturePool.m_Vec2StorageTexBindGroupLayout]
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
			label: "getDivergencePipelineLayout",
			bindGroupLayouts: [this.m_TexturePool.m_Vec2StorageTexBindGroupLayout]
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
			label: "calcPressureStepPipelineLayout",
			bindGroupLayouts: [this.m_TexturePool.m_Vec1StorageTexBindGroupLayout, this.m_TexturePool.m_Vec1StorageTexBindGroupLayout]
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
			label: "projectVelocityPipelineLayout",
			bindGroupLayouts: [this.m_TexturePool.m_Vec1StorageTexBindGroupLayout]
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
			label: "advectVelocityPipelineLayout",
			bindGroupLayouts: [this.m_ParamsBindGroupLayout, this.m_TexturePool.m_SampledTexBindGroupLayout]
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
			label: "displayVec2TexPipelineLayout",
			bindGroupLayouts: [this.m_ParamsBindGroupLayout, this.m_TexturePool.m_Vec2StorageTexBindGroupLayout]
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
			label: "displayVec1TexPipelineLayout",
			bindGroupLayouts: [this.m_ParamsBindGroupLayout, this.m_TexturePool.m_Vec1StorageTexBindGroupLayout]
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
			label: "displayVec1uTexPipelineLayout",
			bindGroupLayouts: [this.m_ParamsBindGroupLayout, this.m_TexturePool.m_Vec1uStorageTexBindGroupLayout]
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

		const drawObstaclesVelocityPipelineLayout = this.m_Device.createPipelineLayout({
			label: "drawObstaclesVelocityPipelineLayout",
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

		const modifyObstaclesVelocityPipelineLayout = this.m_Device.createPipelineLayout({
			label: "modifyObstaclesVelocityPipelineLayout",
			bindGroupLayouts: [
				this.m_ParamsBindGroupLayout,
				this.m_TexturePool.m_Vec2StorageTexBindGroupLayout
			]
		});

		const modifyObstaclesVelocityModule = this.m_Device.createShaderModule({
			code: k_ModifyObstaclesVelocityShader,
		});

		this.m_ModifyObstaclesVelocityPipeline = this.m_Device.createRenderPipeline({
			label: "Modify obstacles velocity pipeline",
			layout: modifyObstaclesVelocityPipelineLayout,
			fragment: {
				module: modifyObstaclesVelocityModule,
				targets: [{
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

		const enforceBoundaryVelocityPipelineLayout = this.m_Device.createPipelineLayout({
			label: "enforceBoundaryVelocityPipelineLayout",
			bindGroupLayouts: [
				this.m_ParamsBindGroupLayout,
				this.m_TexturePool.m_Vec2StorageTexBindGroupLayout,
				this.m_TexturePool.m_Vec2StorageTexBindGroupLayout,
			]
		});

		const enforceBoundaryVelocityModule = this.m_Device.createShaderModule({
			code: k_EnforceBoundaryVelocityShader,
		});

		this.m_EnforceBoundaryVelocityPipeline = this.m_Device.createRenderPipeline({
			label: "Enforce boundary velocity pipeline",
			layout: enforceBoundaryVelocityPipelineLayout,
			fragment: {
				module: enforceBoundaryVelocityModule,
				targets: [{
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
	}

	m_ParamsBuffer = null;
	m_DebugParamsBuffer = null;
	m_Sampler = null;

	m_ObstacleVelocity = null;
	m_ObstacleVelocityView = null;
	m_ObstacleVelocity2 = null;
	m_ObstacleVelocityView2 = null;
	m_ObstacleInstanceBuffer = null;

	createResources(width, height) {
		this.m_SimWidth = width * k_ResolutionScale;
		this.m_SimHeight = height * k_ResolutionScale;

		const pixels = this.m_SimWidth * this.m_SimHeight;
		if (pixels > k_MaxPixels) {
			const ratio = k_MaxPixels / pixels;
			this.m_SimWidth *= ratio;
			this.m_SimHeight *= ratio;
		}

		this.m_SimWidth = Math.round(this.m_SimWidth);
		this.m_SimHeight = Math.round(this.m_SimHeight);

		console.log(`Size: ${this.m_SimWidth}, ${this.m_SimHeight}`);

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

		this.m_Sampler = this.m_Device.createSampler({
			addressModeU: "repeat",
			addressModeV: "repeat",
			magFilter: "linear",
			minFilter: "linear",
		});

		this.m_ObstacleVelocity = this.m_Device.createTexture({
			format: "rg32float",
			size: [this.m_SimWidth, this.m_SimHeight],
			usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING,
			label: "Obstacle Velocity"
		});
		this.m_ObstacleVelocityView = this.m_ObstacleVelocity.createView();

		this.m_ObstacleVelocity2 = this.m_Device.createTexture({
			format: "rg32float",
			size: [this.m_SimWidth, this.m_SimHeight],
			usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING,
			label: "Obstacle Velocity 2"
		});
		this.m_ObstacleVelocityView2 = this.m_ObstacleVelocity2.createView();

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

	m_ObstacleVelocityBindGroup = null;
	m_ObstacleVelocityBindGroup2 = null;

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

		this.m_ObstacleVelocityBindGroup = this.m_Device.createBindGroup({
			layout: this.m_TexturePool.m_Vec2StorageTexBindGroupLayout,
			entries: [
				{
					binding: 0,
					resource: this.m_ObstacleVelocity
				}
			]
		});

		this.m_ObstacleVelocityBindGroup2 = this.m_Device.createBindGroup({
			layout: this.m_TexturePool.m_Vec2StorageTexBindGroupLayout,
			entries: [
				{
					binding: 0,
					resource: this.m_ObstacleVelocity2
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

		if (this.m_SimWidth != this.m_TexturePool.m_SimWidth ||
			this.m_SimHeight != this.m_TexturePool.m_SimHeight) {
			if (this.m_CurrentVelocityTex !== null) {
				this.m_TexturePool.release(this.m_CurrentVelocityTex);
			}
			if (this.m_CurrentPressureTex !== null) {
				this.m_TexturePool.release(this.m_CurrentPressureTex);
			}
			this.m_CurrentVelocityTex = null;
			this.m_CurrentPressureTex = null;

			this.m_TexturePool.setSize(this.m_SimWidth, this.m_SimHeight);
		}

		if (this.m_CurrentVelocityTex == null) {
			this.m_CurrentVelocityTex = this.m_TexturePool.acquire("rg32float");
		}
		if (this.m_CurrentPressureTex == null) {
			this.m_CurrentPressureTex = this.m_TexturePool.acquire("r32float");
		}

		this.updateObstacles();

		let deltaT = (timestep - this.m_PrevTimeStep) / 1000;
		this.m_PrevTimeStep = timestep;

		// Setup frame
		const renderTexture = this.m_Context.getCurrentTexture();
		const renderView = renderTexture.createView();

		const commandEncoder = this.m_Device.createCommandEncoder();

		const obstacleTextures = this.drawObstacles(commandEncoder, deltaT);

		// Set params
		{
			let paramsView = new DataView(new ArrayBuffer(16));

			paramsView.setInt32(0, this.m_SimWidth, true);
			paramsView.setInt32(4, this.m_SimHeight, true);
			paramsView.setFloat32(8, deltaT, true);
			paramsView.setInt32(12, 0, true);

			this.m_Device.queue.writeBuffer(this.m_ParamsBuffer, 0, paramsView);
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

		this.enforceVelocityBoundary(commandEncoder);

		// Diffuse velocity
		{
			for (let i = 0; i < k_VelocityDiffuseSteps; ++i) {
				let nextVelocityTex = this.m_TexturePool.acquire("rg32float");

				const diffuseStepPass = commandEncoder.beginRenderPass({
					colorAttachments: [{
						loadOp: "load",
						storeOp: "store",
						view: nextVelocityTex.m_View,
					}],
				});

				diffuseStepPass.setViewport(0, 0, this.m_SimWidth, this.m_SimHeight, 0, 1);
				diffuseStepPass.setScissorRect(0, 0, this.m_SimWidth, this.m_SimHeight);
				diffuseStepPass.setPipeline(this.m_DiffuseVelocityPipeline);

				diffuseStepPass.setBindGroup(0, this.m_ParamsBindGroup);
				diffuseStepPass.setBindGroup(1, this.m_CurrentVelocityTex.m_StorageBindGroup);

				diffuseStepPass.draw(4);
				diffuseStepPass.end();

				const old = this.m_CurrentVelocityTex;
				this.m_CurrentVelocityTex = nextVelocityTex;
				nextVelocityTex = null;

				this.m_TexturePool.release(old);

				// acts as the swap
				this.enforceVelocityBoundary(commandEncoder);
			}
		}

		this.projectStep(commandEncoder);

		// Advect velocity
		{
			let nextVelocityTex = this.m_TexturePool.acquire("rg32float");

			const advectVelocityPass = commandEncoder.beginRenderPass({
				colorAttachments: [{
					loadOp: "load",
					storeOp: "store",
					view: nextVelocityTex.m_View,
				}],
			});

			advectVelocityPass.setViewport(0, 0, this.m_SimWidth, this.m_SimHeight, 0, 1);
			advectVelocityPass.setScissorRect(0, 0, this.m_SimWidth, this.m_SimHeight);
			advectVelocityPass.setPipeline(this.m_AdvectVelocityPipeline);

			advectVelocityPass.setBindGroup(0, this.m_ParamsBindGroup);
			advectVelocityPass.setBindGroup(1, this.m_CurrentVelocityTex.m_SampledBindGroup);

			advectVelocityPass.draw(4);
			advectVelocityPass.end();

			const old = this.m_CurrentVelocityTex;
			this.m_CurrentVelocityTex = nextVelocityTex;
			nextVelocityTex = null;

			this.m_TexturePool.release(old);
		}

		this.enforceVelocityBoundary(commandEncoder);
		this.projectStep(commandEncoder);
		this.enforceVelocityBoundary(commandEncoder);

		// Display velocity
		{
			const displayPass = commandEncoder.beginRenderPass({
				colorAttachments: [{
					loadOp: "clear",
					storeOp: "store",
					view: renderView,
					clearValue: [0, 0, 0, 1],
				}],
			});

			displayPass.setViewport(0, 0, renderTexture.width, renderTexture.height, 0, 1);
			displayPass.setScissorRect(0, 0, renderTexture.width, renderTexture.height);
			displayPass.setPipeline(this.m_DisplayVec2TexPipeline);

			displayPass.setBindGroup(0, this.m_DebugParamsBindGroup);
			displayPass.setBindGroup(1, this.m_CurrentVelocityTex.m_StorageBindGroup);

			displayPass.draw(4);
			displayPass.end();
		}

		this.m_Device.queue.submit([commandEncoder.finish()]);

		window.requestAnimationFrame(this.frame);

		this.m_TexturePool.checkBalance();
	}

	enforceVelocityBoundary(commandEncoder) {
		let nextVelocityTex = this.m_TexturePool.acquire("rg32float");

		const enforceBoundaryVelPass = commandEncoder.beginRenderPass({
			colorAttachments: [{
				loadOp: "load",
				storeOp: "store",
				view: nextVelocityTex.m_View,
			}],
		});

		enforceBoundaryVelPass.setViewport(0, 0, this.m_SimWidth, this.m_SimHeight, 0, 1);
		enforceBoundaryVelPass.setScissorRect(0, 0, this.m_SimWidth, this.m_SimHeight);
		enforceBoundaryVelPass.setPipeline(this.m_EnforceBoundaryVelocityPipeline);

		enforceBoundaryVelPass.setBindGroup(0, this.m_ParamsBindGroup);
		enforceBoundaryVelPass.setBindGroup(1, this.m_CurrentVelocityTex.m_StorageBindGroup);
		enforceBoundaryVelPass.setBindGroup(2, this.m_ObstacleVelocityBindGroup2);

		enforceBoundaryVelPass.draw(4);
		enforceBoundaryVelPass.end();

		const old = this.m_CurrentVelocityTex;
		this.m_CurrentVelocityTex = nextVelocityTex;
		nextVelocityTex = null;

		this.m_TexturePool.release(old);
	}

	projectStep(commandEncoder) {
		{
			const divergenceTex = this.m_TexturePool.acquire("r32float");

			// Get divergence
			{
				const divergencePass = commandEncoder.beginRenderPass({
					colorAttachments: [{
						loadOp: "load",
						storeOp: "store",
						view: divergenceTex.m_View,
					}],
				});

				divergencePass.setViewport(0, 0, this.m_SimWidth, this.m_SimHeight, 0, 1);
				divergencePass.setScissorRect(0, 0, this.m_SimWidth, this.m_SimHeight);
				divergencePass.setPipeline(this.m_GetDivergencePipeline);

				divergencePass.setBindGroup(0, this.m_CurrentVelocityTex.m_StorageBindGroup);

				divergencePass.draw(4);
				divergencePass.end();
			}

			// Clear pressure
			// const pressureClearPass = commandEncoder.beginRenderPass({
			// 	colorAttachments: [{
			// 		loadOp: "clear",
			// 		storeOp: "store",
			// 		view: this.m_CurrentPressureTex.m_View,
			// 		clearValue: [0, 0, 0, 1]
			// 	}],
			// });
			// pressureClearPass.end();

			// Get pressure
			for (let i = 0; i < k_PressureSteps; ++i) {
				let nextPressureTex = this.m_TexturePool.acquire("r32float");

				const pressureStepPass = commandEncoder.beginRenderPass({
					colorAttachments: [{
						loadOp: "load",
						storeOp: "store",
						view: nextPressureTex.m_View,
					}],
				});

				pressureStepPass.setViewport(0, 0, this.m_SimWidth, this.m_SimHeight, 0, 1);
				pressureStepPass.setScissorRect(0, 0, this.m_SimWidth, this.m_SimHeight);
				pressureStepPass.setPipeline(this.m_CalcPressureStepPipeline);

				pressureStepPass.setBindGroup(0, this.m_CurrentPressureTex.m_StorageBindGroup);
				pressureStepPass.setBindGroup(1, divergenceTex.m_StorageBindGroup);

				pressureStepPass.draw(4);
				pressureStepPass.end();

				const old = this.m_CurrentPressureTex;
				this.m_CurrentPressureTex = nextPressureTex;
				nextPressureTex = null;

				this.m_TexturePool.release(old);
			}

			this.m_TexturePool.release(divergenceTex);
		}

		// Project velocity
		{
			let nextVelocityTex = this.m_TexturePool.acquire("rg32float");

			const projectPass = commandEncoder.beginRenderPass({
				colorAttachments: [{
					loadOp: "load",
					storeOp: "store",
					view: nextVelocityTex.m_View,
				}],
			});

			projectPass.setViewport(0, 0, this.m_SimWidth, this.m_SimHeight, 0, 1);
			projectPass.setScissorRect(0, 0, this.m_SimWidth, this.m_SimHeight);
			projectPass.setPipeline(this.m_ProjectVelocityPipeline);

			projectPass.setBindGroup(0, this.m_CurrentPressureTex.m_StorageBindGroup);

			projectPass.draw(4);
			projectPass.end();

			const old = this.m_CurrentVelocityTex;
			this.m_CurrentVelocityTex = nextVelocityTex;
			nextVelocityTex = null;

			this.m_TexturePool.release(old);
		}
	}

	updateObstacles(timestep) {
		const elements = document.querySelectorAll("fluid-box");
		elements.forEach(el => {
			const rect = el.getBoundingClientRect();

			let exists = this.m_BoxObstaclesDict[el.id] != undefined;
			if (exists) {
				this.m_BoxObstaclesDict[el.id].m_LastPosition = [this.m_BoxObstaclesDict[el.id].m_Position[0], this.m_BoxObstaclesDict[el.id].m_Position[1]];
			} else {
				this.m_BoxObstaclesDict[el.id] = new BoxObstacle(rect.left, rect.top, rect.width, rect.height, timestep);
			}

			let x = (rect.left + k_Inset) / this.m_Canvas.width;
			let y = (rect.top + k_Inset) / this.m_Canvas.height;

			let w = (rect.width - k_Inset * 2) / this.m_Canvas.width;
			let h = (rect.height - k_Inset * 2) / this.m_Canvas.height;

			this.m_BoxObstaclesDict[el.id].m_Position = [x, y];
			this.m_BoxObstaclesDict[el.id].m_Dimensions = [w, h];

			this.m_Timestep = timestep;

			if (!exists) {
				this.m_BoxObstaclesDict[el.id].m_LastPosition = [this.m_BoxObstaclesDict[el.id].m_Position[0], this.m_BoxObstaclesDict[el.id].m_Position[1]];
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

			let deltaX = (value.m_Position[0] - value.m_LastPosition[0]) / deltaT;
			let deltaY = (value.m_Position[1] - value.m_LastPosition[1]) / deltaT;

			if (Math.abs(deltaX) > k_TeleportThreshold || deltaT == 0) deltaX = 0;
			if (Math.abs(deltaY) > k_TeleportThreshold || deltaT == 0) deltaY = 0;

			instanceData.setFloat32(offset + 16, deltaX, true);
			instanceData.setFloat32(offset + 20, deltaY, true);

			offset += k_ObstacleInstanceSize;
		});

		this.m_Device.queue.writeBuffer(this.m_ObstacleInstanceBuffer, 0, instanceData);

		// Draw to velocity
		const velPass = commandEncoder.beginRenderPass({
			colorAttachments: [{
				loadOp: "clear",
				storeOp: "store",
				view: this.m_ObstacleVelocityView,
				clearValue: [9999, 9999, 0, 1]
			}],
		});

		velPass.setViewport(0, 0, this.m_SimWidth, this.m_SimHeight, 0, 1);
		velPass.setScissorRect(0, 0, this.m_SimWidth, this.m_SimHeight);
		velPass.setPipeline(this.m_DrawObstaclesVelocityPipeline);
		velPass.setVertexBuffer(0, this.m_ObstacleInstanceBuffer);

		velPass.draw(4, obstacles.length);

		velPass.end();

		// Velocity edge detection
		const edgePass = commandEncoder.beginRenderPass({
			colorAttachments: [{
				loadOp: "clear",
				storeOp: "store",
				view: this.m_ObstacleVelocityView2,
				clearValue: [0, 0, 0, 1]
			}],
		});

		edgePass.setViewport(0, 0, this.m_SimWidth, this.m_SimHeight, 0, 1);
		edgePass.setScissorRect(0, 0, this.m_SimWidth, this.m_SimHeight);
		edgePass.setPipeline(this.m_ModifyObstaclesVelocityPipeline);

		edgePass.setBindGroup(0, this.m_ParamsBindGroup);
		edgePass.setBindGroup(1, this.m_ObstacleVelocityBindGroup);

		edgePass.draw(4);

		edgePass.end();

		return {};
	}
}