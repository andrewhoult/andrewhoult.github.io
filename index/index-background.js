//
// Thanks to Jos Stam's paper "Real-Time Fluid Dynamics for Games"
// https://www.dgp.toronto.edu/public_user/stam/reality/Research/pdf/GDC03.pdf
//
// I've adapted the algorithm presented there to run on the GPU by using Jacobian
// solvers with swapping textures.
//

const k_ResolutionScale = 0.25;

const k_VelocityDiffuseSteps = 20;

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
override viscosity: f32 = 20; // m^2/s

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
override viscosity: f32 = 0.000001488; // m^2/s

@group(0) @binding(0) var v_tex : texture_storage_2d<rg32float, read>;

@fragment
fn fragment_main(@builtin(position) fragCoord: vec4<f32>) -> @location(0) f32 {
  let texelCoord 	= vec2<i32>(fragCoord.xy);
  let leftCoord 	= texelCoord + vec2<i32>(-1,  0);
  let rightCoord 	= texelCoord + vec2<i32>( 1,  0);
  let downCoord 	= texelCoord + vec2<i32>( 0, -1);
  let upCoord 		= texelCoord + vec2<i32>( 0,  1);

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
  let leftCoord 	= texelCoord + vec2<i32>(-1,  0);
  let rightCoord 	= texelCoord + vec2<i32>( 1,  0);
  let downCoord 	= texelCoord + vec2<i32>( 0, -1);
  let upCoord 		= texelCoord + vec2<i32>( 0,  1);

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
  let leftCoord 	= texelCoord + vec2<i32>(-1,  0);
  let rightCoord 	= texelCoord + vec2<i32>( 1,  0);
  let downCoord 	= texelCoord + vec2<i32>( 0, -1);
  let upCoord 		= texelCoord + vec2<i32>( 0,  1);

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
@group(1) @binding(0) var v0_tex : texture_2d<vec2<f32>>;
@group(1) @binding(1) var v0_sampler : sampler;

@fragment
fn fragment_main(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec2<f32> {
  let uv = fragCoord.xy / vec2<f32>(params.resolution);
  let v0 = textureSample(v0_tex, v0_sampler, uv).xy;
  
  let back = -v0 * params.dT / vec2<f32>(params.resolution);
  let takeUV = uv - back;

  let v1 = textureSample(v0_tex, v0_sampler, takeUV).xy;

  return v1;
}
`;

// Render velocity as colour info for debugging
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
  
  return vec4<f32>(v, 0, 1);
}
`;

class BackgroundRenderer {
	m_Canvas = null;
	m_Adapter = null;
	m_Device = null;
	m_Context = null;

	m_IsReady = false;

	m_RenderingA = true;
	m_SimWidth = 100;
	m_SimHeight = 100;

	m_MouseU = 0;
	m_MouseV = 0;
	m_IsMouseDown = false;

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

		this.m_Adapter = await navigator.gpu.requestAdapter();
		if (!this.m_Adapter) {
			throw Error("Couldn't request WebGPU adapter.");
		}

		if (!this.m_Adapter.features.has("float32-blendable")) {
			throw Error("float32-blendable not supported.");
		}

		this.m_Device = await this.m_Adapter.requestDevice({
			requiredFeatures: ["float32-blendable"]
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

				canvas.width = Math.floor(width * window.devicePixelRatio);
				canvas.height = Math.floor(height * window.devicePixelRatio);

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

	m_AddForcesPipeline = null;
	m_ParamsBindGroupLayout = null;
	m_Vec2StorageTexBindGroupLayout = null;
	m_Vec1StorageTexBindGroupLayout = null;
	m_SampledTexBindGroupLayout = null;

	m_AddForcesPipeline = null;
	m_DiffuseVelocityPipeline = null;
	m_GetDivergencePipeline = null;
	m_CalcPressureStepPipeline = null;
	m_ProjectVelocityPipeline = null;
	m_DisplayVec2TexPipeline = null;

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
	}

	m_ParamsBuffer = null;
	m_DebugParamsBuffer = null;
	m_ForcesTex = null;
	m_ForcesTexView = null;
	m_VelocityTexA = null;
	m_VelocityTexViewA = null;
	m_VelocityTexB = null;
	m_VelocityTexViewB = null;

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
			label: "Velocity Texture A"
		});
		this.m_VelocityTexViewA = this.m_VelocityTexA.createView();

		this.m_VelocityTexB = this.m_Device.createTexture({
			format: "rg32float",
			size: [this.m_SimWidth, this.m_SimHeight],
			usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
			label: "Velocity Texture B"
		});
		this.m_VelocityTexViewB = this.m_VelocityTexB.createView();
	}

	m_ParamsBindGroup = null;
	m_DebugParamsBindGroup = null;
	m_ForcesStorageTexBindGroup = null;
	m_VelocityStorageTexBindGroupA = null;
	m_VelocityStorageTexBindGroupB = null;

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
	}

	async frame(timestep) {
		if (!this.m_IsReady) {
			this.m_PrevTimeStep = timestep;
			window.requestAnimationFrame(this.frame);
			return;
		}

		let deltaT = (timestep - this.m_PrevTimeStep) / 1000;
		this.m_PrevTimeStep = timestep;

		// Setup frame
		const renderTexture = this.m_Context.getCurrentTexture();
		const renderView = renderTexture.createView();

		const commandEncoder = this.m_Device.createCommandEncoder();

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
			let forcesView = new DataView(new ArrayBuffer(8));
			forcesView.setFloat32(0, 0, true);
			forcesView.setFloat32(4, 500, true);

			const x = Math.min(Math.max(this.m_MouseU * this.m_SimWidth, 0), this.m_SimWidth - 1);
			const y = Math.min(Math.max(this.m_MouseV * this.m_SimHeight, 0), this.m_SimHeight - 1);

			this.m_Device.queue.writeTexture(
				{
					origin: [x, y, 0],
					texture: this.m_ForcesTex
				},
				forcesView,
				{},
				{
					width: 1,
					height: 1,
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
					view: this.m_RenderingA ? this.m_VelocityTexViewA : this.m_VelocityTexViewB,
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
			this.m_RenderingA = !this.m_RenderingA;

			const diffuseStepPass = commandEncoder.beginRenderPass({
				colorAttachments: [{
					loadOp: "load",
					storeOp: "store",
					view: this.m_RenderingA ? this.m_VelocityTexViewA : this.m_VelocityTexViewB,
				}],
			});

			diffuseStepPass.setViewport(0, 0, this.m_SimWidth, this.m_SimHeight, 0, 1);
			diffuseStepPass.setScissorRect(0, 0, this.m_SimWidth, this.m_SimHeight);
			diffuseStepPass.setPipeline(this.m_DiffuseVelocityPipeline);

			diffuseStepPass.setBindGroup(0, this.m_ParamsBindGroup);
			diffuseStepPass.setBindGroup(1, this.m_RenderingA ? this.m_VelocityStorageTexBindGroupB : this.m_VelocityStorageTexBindGroupA);

			diffuseStepPass.draw(4);
			diffuseStepPass.end();
		}

		// Display velocity
		{
			const displayVelocityPass = commandEncoder.beginRenderPass({
				colorAttachments: [{
					loadOp: "clear",
					storeOp: "store",
					view: renderView,
					clearValue: [0, 0, 0, 1],
				}],
			});

			displayVelocityPass.setViewport(0, 0, renderTexture.width, renderTexture.height, 0, 1);
			displayVelocityPass.setScissorRect(0, 0, renderTexture.width, renderTexture.height);
			displayVelocityPass.setPipeline(this.m_DisplayVec2TexPipeline);

			displayVelocityPass.setBindGroup(0, this.m_DebugParamsBindGroup);
			displayVelocityPass.setBindGroup(1, this.m_RenderingA ? this.m_VelocityStorageTexBindGroupA : this.m_VelocityStorageTexBindGroupB);

			displayVelocityPass.draw(4);
			displayVelocityPass.end();
		}

		this.m_Device.queue.submit([commandEncoder.finish()]);

		window.requestAnimationFrame(this.frame);
	}
}