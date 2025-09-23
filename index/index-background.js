//
// Thanks to Jos Stam's paper "Real-Time Fluid Dynamics for Games"
// https://www.dgp.toronto.edu/public_user/stam/reality/Research/pdf/GDC03.pdf
//
// I've adapted the algorithm presented there to run on the GPU by using Jacobian
// solvers with swapping textures.
//

const k_FramesInFlight = 1;

let g_BackgroundRenderer;

document.addEventListener("DOMContentLoaded", () => {
	let canvas = document.getElementById("funny-background-canvas");
	g_BackgroundRenderer = new BackgroundRenderer();
	g_BackgroundRenderer.init(canvas);
});

// Add forces from force texture to velocity texture
// Additive blending required
// Renders to v0_tex
const k_AddForcesShader = `
@group(0) @binding(0) var dt : f32;
@group(0) @binding(1) var sources : texture_storage_2d<rg32float, read>;

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

@fragment
fn fragment_main(@builtin(frag_coord) fragCoord: vec4<f32>) -> @location(0) vec2<f32> {
  let texelCoord	= vec2<i32>(fragCoord.xy);
  let source 		= textureLoad(sources, texelCoord, 0).xy;
  return dt * source.xy; // Additive blending required
}
`;

// Jacobian solver step to diffuse velocity
// Renders to v1_tex, swap each iteration
const k_DiffuseVelocityStepShader = `
override viscosity: f32 = 0.000001488; // m^2/s

@group(0) @binding(0) var dt : f32;
@group(0) @binding(1) var v0_tex : texture_storage_2d<rg32float, read>;

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

@fragment
fn fragment_main(@builtin(frag_coord) fragCoord: vec4<f32>) -> @location(0) vec2<f32> {
  let texelCoord 	= vec2<i32>(fragCoord.xy);
  let leftCoord 	= texelCoord + vec2<i32>(-1,  0);
  let rightCoord 	= texelCoord + vec2<i32>( 1,  0);
  let downCoord 	= texelCoord + vec2<i32>( 0, -1);
  let upCoord 		= texelCoord + vec2<i32>( 0,  1);

  let a = viscosity * dt; // grid size is 1

  let v0 		= textureLoad(v0_tex, texelCoord, 0).xy;
  let v0Left 	= textureLoad(v0_tex, leftCoord,  0).xy;
  let v0Right 	= textureLoad(v0_tex, rightCoord, 0).xy;
  let v0Down 	= textureLoad(v0_tex, downCoord,  0).xy;
  let v0Up 		= textureLoad(v0_tex, upCoord,    0).xy;

  let v1 = (v0 + a * (v0Left + v0Right + v0Down + v0Up)) / (1.0 + 4.0 * a);

  return v1;
}
`;

// Renders to divergence tex
const k_GetDivergenceShader = `
override viscosity: f32 = 0.000001488; // m^2/s

@group(0) @binding(0) var dt : f32;
@group(0) @binding(1) var v_tex : texture_storage_2d<rg32float, read>;

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

@fragment
fn fragment_main(@builtin(frag_coord) fragCoord: vec4<f32>) -> @location(0) f32 {
  let texelCoord 	= vec2<i32>(fragCoord.xy);
  let leftCoord 	= texelCoord + vec2<i32>(-1,  0);
  let rightCoord 	= texelCoord + vec2<i32>( 1,  0);
  let downCoord 	= texelCoord + vec2<i32>( 0, -1);
  let upCoord 		= texelCoord + vec2<i32>( 0,  1);

  let vLeft 	= textureLoad(v0_tex, leftCoord,  0).xy;
  let vRight 	= textureLoad(v0_tex, rightCoord, 0).xy;
  let vDown 	= textureLoad(v0_tex, downCoord,  0).xy;
  let vUp 		= textureLoad(v0_tex, upCoord,    0).xy;

  let div = -0.5 * (vRight.x - vLeft.x + vUp.y - vDown.y);

  return div;
}
`;

// Jacobian solver step to find pressure
// Renders to p1_tex, swap each iteration
const k_CalcPressureStepShader = `
@group(0) @binding(0) var dt : f32;
@group(0) @binding(1) var p0_tex : texture_storage_2d<r32float, read>;
@group(0) @binding(2) var div_tex : texture_storage_2d<r32float, read>;

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

@fragment
fn fragment_main(@builtin(frag_coord) fragCoord: vec4<f32>) -> @location(0) f32 {
  let texelCoord 	= vec2<i32>(fragCoord.xy);
  let leftCoord 	= texelCoord + vec2<i32>(-1,  0);
  let rightCoord 	= texelCoord + vec2<i32>( 1,  0);
  let downCoord 	= texelCoord + vec2<i32>( 0, -1);
  let upCoord 		= texelCoord + vec2<i32>( 0,  1);

  let div 		= textureLoad(div_tex, texelCoord, 0).xy;
  let p0Left 	= textureLoad(v0_tex,  leftCoord,  0).xy;
  let p0Right 	= textureLoad(v0_tex,  rightCoord, 0).xy;
  let p0Down 	= textureLoad(v0_tex,  downCoord,  0).xy;
  let p0Up 		= textureLoad(v0_tex,  upCoord,    0).xy;

  let p1 = 0.25 * (div + (p0Left + p0Right + p0Down + p0Up));

  return v1;
}
`;

// Project velocity, make it mass conserving
// Renders to v_tex. Requires additive blending.
const k_ProjectVelocityShader = `
@group(0) @binding(0) var p_tex : texture_storage_2d<r32float, read>;

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

@fragment
fn fragment_main(@builtin(frag_coord) fragCoord: vec4<f32>) -> @location(0) vec2<f32> {
  let texelCoord 	= vec2<i32>(fragCoord.xy);
  let leftCoord 	= texelCoord + vec2<i32>(-1,  0);
  let rightCoord 	= texelCoord + vec2<i32>( 1,  0);
  let downCoord 	= texelCoord + vec2<i32>( 0, -1);
  let upCoord 		= texelCoord + vec2<i32>( 0,  1);

  let pLeft 	= textureLoad(v0_tex,  leftCoord,  0).xy;
  let pRight 	= textureLoad(v0_tex,  rightCoord, 0).xy;
  let pDown 	= textureLoad(v0_tex,  downCoord,  0).xy;
  let pUp 		= textureLoad(v0_tex,  upCoord,    0).xy;

  let p1 = 0.25 * (div + (p0Left + p0Right + p0Down + p0Up));

  let sub = 0.5 * vec2<f32>(pRight.x - pLeft.x, pUp.y - pDown.y);

  return sub; // Requires additive blending
}
`;

class BackgroundRenderer {
	m_Canvas = null;
	m_Adapter = null;
	m_Device = null;
	m_Context = null;

	m_SharedReadBindGroupLayout = null;
	m_TestShaderPipeline = null;

	m_SizeDirty = true;
	m_FrameData = [];
	m_FrameNumber = 0;

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

		this.m_Device = await this.m_Adapter.requestDevice();
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

		this.compileShaders();
		this.createResources();

		const resizeObserver = new ResizeObserver(entries => {
			for (const entry of entries) {
				const width = entry.contentRect.width;
				const height = entry.contentRect.height;

				canvas.width = Math.floor(width * window.devicePixelRatio);
				canvas.height = Math.floor(height * window.devicePixelRatio);

				console.log("Resize: " + this.m_Canvas.width + ", " + this.m_Canvas.height);
				this.m_SizeDirty = true;
			}
		});

		resizeObserver.observe(this.m_Canvas);
		window.requestAnimationFrame(this.frame);
	}

	compileShaders() {
		this.m_SharedReadBindGroupLayout = this.m_Device.createBindGroupLayout({
			entries: [
				{
					binding: 0,
					visibility: GPUShaderStage.FRAGMENT,
					texture: {}
				},
			]
		});

		const sharedPipelineLayout = this.m_Device.createPipelineLayout({
			bindGroupLayouts: [this.m_SharedReadBindGroupLayout]
		});

		const testShaderModule = this.m_Device.createShaderModule({
			code: k_TestShader,
		});

		this.m_TestShaderPipeline = this.m_Device.createRenderPipeline({
			layout: sharedPipelineLayout,
			fragment: {
				module: testShaderModule,
				targets: [{
					blend: {
						color: {
							srcFactor: "src-alpha",
							dstFactor: "one-minus-src-alpha",
							operation: "add"
						},
						alpha: {
							srcFactor: "one",
							dstFactor: "one-minus-src-alpha",
							operation: "add"
						}
					},
					format: navigator.gpu.getPreferredCanvasFormat()
				}],
			},
			vertex: {
				module: testShaderModule,
			}
		});
	}

	async createResources() {
		await this.m_Device.queue.onSubmittedWorkDone();

		this.m_FrameData = new Array(k_FramesInFlight);
		for (let i = 0; i < k_FramesInFlight; ++i) {
			this.m_FrameData[i] = new FrameData(this);
		}
		this.m_FrameNumber = 0;
	}

	async frame(timestep) {
		if (this.m_SizeDirty) {
			this.m_SizeDirty = false;
		}

		const renderTexture = this.m_Context.getCurrentTexture();
		const renderView = renderTexture.createView();

		const commandEncoder = this.m_Device.createCommandEncoder();
		const renderPass = commandEncoder.beginRenderPass({
			colorAttachments: [{
				clearValue: [0, 0, 0, 0],
				loadOp: "clear",
				storeOp: "store",
				view: renderView,
			}],
		});

		const fd = this.m_FrameData[this.m_FrameNumber];

		renderPass.setViewport(0, 0, 100, 100, 0, 1);
		renderPass.setScissorRect(0, 0, 100, 100);
		renderPass.setPipeline(this.m_TestShaderPipeline);

		renderPass.setPipeline(this.m_TestShaderPipeline);

		//renderPass.setBindGroup(0, fd.m_ColorBindGroup);
		//renderPass.draw(3);

		renderPass.end();

		// commandEncoder.copyTextureToTexture(
		// 	{
		// 		texture: fd.m_SimTexture,
		// 		origin: [0, 0, 0],
		// 	},
		// 	{
		// 		texture: renderTexture,
		// 		origin: [0, 0, 0],
		// 	},
		// 	[100, 100]
		// );

		this.m_Device.queue.submit([commandEncoder.finish()]);

		this.m_FrameNumber = (this.m_FrameNumber + 1) % k_FramesInFlight;
		window.requestAnimationFrame(this.frame);
	}
}