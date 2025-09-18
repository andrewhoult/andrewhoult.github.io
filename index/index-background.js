//
// Thanks to Jos Stam's paper "Real-Time Fluid Dynamics for Games"
// https://www.dgp.toronto.edu/public_user/stam/reality/Research/pdf/GDC03.pdf
//
// I've adapted the algorithm presented there to run using shaders.
//

const k_FramesInFlight = 1;

let g_BackgroundRenderer;

document.addEventListener("DOMContentLoaded", () => {
	let canvas = document.getElementById("funny-background-canvas");
	g_BackgroundRenderer = new BackgroundRenderer();
	g_BackgroundRenderer.init(canvas);
});

// Additive blending required
const k_AddForcesShader = `
@group(0) @binding(0) var dt : f32;
@group(0) @binding(1) var sources : texture_2d<f32>;

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
  let texelCoord = vec2<i32>(i32(fragCoord.x), i32(fragCoord.y));
  let source = textureLoad(sources, texelCoord, 0);
  return dt * source.r; // Additive blending required
}
`;

// One step of the diffusion process
const k_DiffuseStepShader = `
@group(0) @binding(0) var scaledDiffusion : f32; // Diffusion scaled by dt.
@group(0) @binding(1) var inputTex : texture_2d<f32>;

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

fn safeLoad(tex: texture_2d<f32>, coord: vec2<i32>) -> f32 {
  let size = textureDimensions(tex, 0);
  let clamped = clamp(coord, vec2<i32>(0), vec2<i32>(i32(size.x - 1), i32(size.y - 1)));
  return textureLoad(tex, clamped, 0).r;
}

@fragment
fn fragment_main(@builtin(frag_coord) fragCoord: vec4<f32>) -> @location(0) f32 {
  let texelCoord = vec2<i32>(i32(fragCoord.x), i32(fragCoord.y));
  let input = safeLoad(inputTex, texelCoord, 0);
  let inputLeft = safeLoad(inputTex, texelCoord + vec2<i32>(-1, 0), 0);
  let inputRight = safeLoad(inputTex, texelCoord + vec2<i32>(1, 0), 0);
  let inputUp = safeLoad(inputTex, texelCoord + vec2<i32>(0, 1), 0);
  let inputDown = safeLoad(inputTex, texelCoord + vec2<i32>(0, -1), 0);
  return (input + scaledDiffusion * (inputLeft + inputRight + inputUp + inputDown)) / (1 + 4 * scaledDiffusion);
}
`;

// Advection
const k_AdvectionShader = `
@group(0) @binding(0) var dt : f32;
@group(0) @binding(1) var d0 : texture_2d<f32>; // Previous density, or whatever else we're advecting
@group(0) @binding(2) var velX : texture_2d<f32>;
@group(0) @binding(3) var velY : texture_2d<f32>;

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

fn safeLoad(tex: texture_2d<f32>, coord: vec2<i32>) -> f32 {
  let size = textureDimensions(tex, 0);
  let clamped = clamp(coord, vec2<i32>(0), vec2<i32>(i32(size.x - 1), i32(size.y - 1)));
  return textureLoad(tex, clamped, 0).r;
}

@fragment
fn fragment_main(@builtin(frag_coord) fragCoord: vec4<f32>) -> @location(0) f32 {
  let texelCoord = vec2<i32>(i32(fragCoord.x), i32(fragCoord.y));
  let size: vec2<u32> = textureDimensions(velX, 0);

  var x: f32 = fragCoord.x - dt * safeLoad(velX, texelCoord);
  var y: f32 = fragCoord.y - dt * safeLoad(velY, texelCoord);

  // Because we're simulating from midpoints
  x = clamp(x, 0.5, size - 0.5);
  y = clamp(y, 0.5, size - 0.5);

  let i0: u32 = (u32)x;
  let i1: u32 = i0 + 1;

  let j0: u32 = (u32)y;
  let j1: u32 = j0 + 1;

  let s1 = x - i0;
  let s0 = 1 - s1;
  let t1 = y - j0;
  let t0 = 1 - t1;

  float d0i0j0 = safeLoad(d0, vec2<i32>(i0, j0));
  float d0i0j1 = safeLoad(d0, vec2<i32>(i0, j1));
  float d0i1j0 = safeLoad(d0, vec2<i32>(i1, j0));
  float d0i1j1 = safeLoad(d0, vec2<i32>(i1, j1));

  float d = s0 * (t0 * d0i0j0 + t1 * d0i0j1) + 
			s1 * (t0 * d0i1j0 + t1 * d0i1j1);
  return d;
}
`;

class FrameData {
	m_VelocityPressureTexture = null;
	m_ColorTexture = null;

	m_VelocityPressureTextureView = null;
	m_ColorTextureView = null;

	m_VelocityPressureBindGroup = null;
	m_ColorBindGroup = null;

	constructor(renderer) {
		// TODO: Revisit usage when done
		this.m_VelocityPressureTexture = renderer.m_Device.createTexture({
			dimension: "2d",
			format: "rgba32float",
			size: [100, 100],
			usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
		});

		this.m_ColorTexture = renderer.m_Device.createTexture({
			dimension: "2d",
			format: "rgba32float",
			size: [100, 100],
			usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
		});

		this.m_VelocityPressureTextureView = this.m_VelocityPressureTexture.createView();
		this.m_ColorTextureView = this.m_ColorTexture.createView();

		this.m_VelocityPressureBindGroup = renderer.m_Device.createBindGroup({
			entries: [
				{
					binding: 0,
					resource: this.m_VelocityPressureTextureView,
				}
			],
			layout: renderer.m_SharedBindGroupLayout
		});

		this.m_ColorBindGroup = renderer.m_Device.createBindGroup({
			entries: [
				{
					binding: 0,
					resource: this.m_ColorTextureView,
				}
			],
			layout: renderer.m_SharedBindGroupLayout
		});
	}
}

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
		//await this.createFrameData();

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

		this.m_SharedWriteBindGroupLayout = this.m_Device.createBindGroupLayout({
			entries: [
				{
					binding: 0,
					visibility: GPUShaderStage.FRAGMENT,
					storageTexture: {
						access: "read-write",
						format: "rg32float"
					}
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

	async createFrameData() {
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