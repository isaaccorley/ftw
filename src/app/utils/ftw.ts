"use client";

/**
 * Client-side utilities for running ONNX models in the browser.
 * Uses onnxruntime-web for browser-based inference.
 */

import type { InferenceSession, Tensor } from "onnxruntime-web";
import { upsamplePatch } from "./common";
import type { ModelOption } from "./model-loader";
import { MODEL_OPTIONS } from "./model-loader";

// Try to suppress ONNX Runtime warnings early
try {
  // Set environment variables before importing ONNX Runtime
  if (typeof window !== "undefined") {
    // This might help suppress warnings in the browser
    (window as unknown as Record<string, unknown>).ORT_LOG_LEVEL = 3; // Error level only
  }
} catch {
  // Ignore if not available
}

type OrtModule = typeof import("onnxruntime-web");

let ortModulePromise: Promise<OrtModule> | null = null;

const getOrtModule = (): Promise<OrtModule> => {
  if (!ortModulePromise) {
    ortModulePromise = (async () => {
      const ort = await import("onnxruntime-web");

      // Try multiple approaches to suppress warnings
      try {
        // Method 1: Set log level
        if (ort.env && typeof ort.env.logLevel !== "undefined") {
          ort.env.logLevel = "error";
        }

        // Method 2: Set log severity level
        if (ort.env && typeof ort.env.logSeverityLevel !== "undefined") {
          ort.env.logSeverityLevel = 3; // Error level only
        }

        // Method 3: Try to set verbose level
        if (ort.env && typeof ort.env.verbose !== "undefined") {
          ort.env.verbose = false;
        }
      } catch (e) {
        // Ignore if log level settings are not available
        console.warn("Could not set ONNX Runtime log level:", e);
      }

      return ort;
    })();
  }
  return ortModulePromise;
};

export interface FieldOnnxModelOptions {
  /**
   * URL or ArrayBuffer of the ONNX model to load.
   */
  modelPath: string | ArrayBuffer;
  /**
   * Optional execution providers to try, in order of preference.
   * Defaults to ['webgpu', 'webgl', 'wasm'].
   */
  executionProviders?: string[];
  /**
   * Optional log severity level (0=verbose, 1=info, 2=warning, 3=error, 4=fatal).
   * Defaults to 2 (warnings and errors only).
   */
  logSeverityLevel?: number;
}

export interface FieldOnnxInferenceInput {
  /**
   * Flattened Float32 pixel data in RGBN order (4 channels).
   */
  data: Float32Array;
  /**
   * Width of the raster (pixels).
   */
  width: number;
  /**
   * Height of the raster (pixels).
   */
  height: number;
  /**
   * Optional divisor used to normalise reflectance values (defaults to 4000).
   */
  normalization?: number;
  /**
   * If true, the tensor will be converted from NHWC to NCHW (default true).
   */
  nchw?: boolean;
}

export interface FieldOnnxModelResult {
  session: InferenceSession;
  outputs: Record<string, Tensor>;
}

// Try WebGPU first, then WebGL, then WASM as fallback
// Some operations will always run on CPU for performance reasons
const DEFAULT_EXECUTION_PROVIDERS = ["webgpu", "webgl", "wasm"];

const loadModelSession = async (options: FieldOnnxModelOptions): Promise<InferenceSession> => {
  const ort = await getOrtModule();

  const sessionOptions = {
    executionProviders: options.executionProviders ?? DEFAULT_EXECUTION_PROVIDERS,
    logSeverityLevel: (options.logSeverityLevel ?? 3) as 0 | 1 | 2 | 3 | 4,
    // Enable memory optimizations
    enableCpuMemArena: true,
    enableMemPattern: true,
  };

  // Handle both string URLs and ArrayBuffer model data
  let session: InferenceSession;
  if (typeof options.modelPath === "string") {
    console.log("Creating ONNX session from URL:", options.modelPath);
    session = await ort.InferenceSession.create(options.modelPath, sessionOptions);
  } else {
    console.log("Creating ONNX session from ArrayBuffer, size:", options.modelPath.byteLength);
    session = await ort.InferenceSession.create(new Uint8Array(options.modelPath), sessionOptions);
  }

  return session;
};

const normalisePixelData = (
  input: Float32Array,
  width: number,
  height: number,
  normalization: number,
): Float32Array => {
  const inv = normalization > 0 ? 1 / normalization : 1 / 4000;
  const length = width * height * 4;
  const output = new Float32Array(length);
  for (let i = 0; i < length; i += 1) {
    const value = input[i] * inv;
    output[i] = Math.min(Math.max(value, 0), 1);
  }
  return output;
};

const nhwcToNchw = (
  input: Float32Array,
  width: number,
  height: number,
  channels = 4,
): Float32Array => {
  const output = new Float32Array(channels * height * width);
  let dstIndex = 0;
  for (let c = 0; c < channels; c += 1) {
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const srcIndex = (y * width + x) * channels + c;
        output[dstIndex] = input[srcIndex];
        dstIndex += 1;
      }
    }
  }
  return output;
};

export const createFieldOnnxSession = loadModelSession;

export const runFieldOnnxInference = async (
  session: InferenceSession,
  input: FieldOnnxInferenceInput,
): Promise<FieldOnnxModelResult> => {
  const ort = await getOrtModule();

  // Validate session
  if (!session) {
    throw new Error("ONNX session is null or undefined. Session creation may have failed.");
  }

  if (input.data.length !== input.width * input.height * 4) {
    throw new Error(
      `Expected RGBN data length ${input.width * input.height * 4}, received ${input.data.length}`,
    );
  }

  const normalizationDivisor =
    input.normalization && input.normalization > 0 ? input.normalization : 4000;

  const normalised = normalisePixelData(
    input.data,
    input.width,
    input.height,
    normalizationDivisor,
  );

  const tensorData =
    input.nchw === false ? normalised : nhwcToNchw(normalised, input.width, input.height);

  const tensorShape =
    input.nchw === false ? [1, input.height, input.width, 4] : [1, 4, input.height, input.width];

  if (!session.inputNames || session.inputNames.length === 0) {
    throw new Error("ONNX session has no input names. Session may not be properly initialized.");
  }

  const feeds: Record<string, Tensor> = {};
  const inputName = session.inputNames[0];

  // For now, use float32 for all models to avoid data type conversion issues
  // Many ONNX models can accept float32 input even if they're FP16 models
  const dataType = "float32";
  feeds[inputName] = new ort.Tensor(dataType, tensorData, tensorShape);

  const outputs = await session.run(feeds);
  return {
    session,
    outputs,
  };
};

export interface OnnxSegmentationResult {
  mask: number[];
  width: number;
  height: number;
  classes: string[];
  classCounts: number[];
  confidences: number[];
}

export const runSegmentationInference = async (
  session: InferenceSession,
  input: FieldOnnxInferenceInput,
  classes: string[],
  scoreThreshold: number = 0.5,
): Promise<OnnxSegmentationResult> => {
  const result = await runFieldOnnxInference(session, input);

  const outputNames = session.outputNames ?? Object.keys(result.outputs);
  const firstOutputName = outputNames[0];
  const tensor = result.outputs[firstOutputName];

  if (!tensor) {
    throw new Error("ONNX model inference returned no outputs.");
  }

  if (!tensor.dims || tensor.dims.length < 3) {
    throw new Error(`Unexpected output tensor shape: ${tensor.dims}`);
  }

  const [batch, channels, outHeight, outWidth] =
    tensor.dims.length === 4 ? tensor.dims : [1, tensor.dims[0], tensor.dims[1], tensor.dims[2]];

  if (batch !== 1) {
    throw new Error(`Expected batch size 1, received ${batch}`);
  }

  const values = tensor.data as Float32Array | Float64Array | number[];
  const numPixels = outHeight * outWidth;
  const mask = new Uint8Array(numPixels);
  const counts = new Array(channels).fill(0);
  const confidences = new Float32Array(numPixels);

  for (let idx = 0; idx < numPixels; idx += 1) {
    // stable softmax per-pixel
    let maxLogit = -Infinity;
    for (let c = 0; c < channels; c += 1) {
      const v = values[c * numPixels + idx] as number;
      if (v > maxLogit) maxLogit = v;
    }
    let sumExp = 0;
    const probs = new Float32Array(channels);
    for (let c = 0; c < channels; c += 1) {
      const ex = Math.exp((values[c * numPixels + idx] as number) - maxLogit);
      probs[c] = ex;
      sumExp += ex;
    }
    let bestClass = 0;
    let bestProb = 0;
    for (let c = 0; c < channels; c += 1) {
      const p = probs[c] / (sumExp || 1);
      if (p > bestProb) {
        bestProb = p;
        bestClass = c;
      }
    }
    const finalClass = bestProb >= scoreThreshold ? bestClass : 0;
    mask[idx] = finalClass;
    confidences[idx] = bestProb;
    counts[finalClass] += 1;
  }

  return {
    mask: Array.from(mask),
    width: outWidth,
    height: outHeight,
    classes,
    classCounts: counts,
    confidences: Array.from(confidences),
  };
};

// FTW-specific types and functions
export type OnnxSegmentationRequest = {
  modelId: string;
  width: number;
  height: number;
  normalization: number;
  data: ArrayBuffer;
  scoreThreshold: number;
};

export type OnnxPatchRequest = {
  modelId: string;
  patchData: ArrayBuffer;
  patchWidth: number;
  patchHeight: number;
  normalization: number;
  scoreThreshold: number;
};

export type OnnxSegmentationResponse = {
  mask: number[];
  width: number;
  height: number;
  classes: string[];
  classCounts: number[];
};

export type OnnxPatchResponse = {
  mask: number[];
  width: number;
  height: number;
  classes: string[];
  confidences: number[];
};

const getModelOption = (modelId: string): ModelOption => {
  const option = MODEL_OPTIONS.find((candidate) => candidate.id === modelId);
  if (!option) {
    throw new Error(`Unknown model id "${modelId}"`);
  }
  if (option.engine !== "onnx") {
    throw new Error(`Model "${option.name}" is not an ONNX segmentation model.`);
  }
  return option;
};

export const runSegmentationInferenceClient = async (
  session: InferenceSession,
  payload: OnnxSegmentationRequest,
): Promise<OnnxSegmentationResponse> => {
  const option = getModelOption(payload.modelId);
  const floatData = new Float32Array(payload.data);

  const result = await runSegmentationInference(
    session,
    {
      data: floatData,
      width: payload.width,
      height: payload.height,
      normalization: payload.normalization,
    },
    option.classes ?? [],
    payload.scoreThreshold,
  );

  return result;
};

// ─── YOLO-seg helpers ────────────────────────────────────────────────────────

const YOLO_INPUT_SIZE = 512;
// YOLO_PROTO_SIZE removed — actual proto dims are read from out1.dims at runtime
const YOLO_NUM_PROTOS = 32;

/**
 * Convert a Float32Array to a Uint16Array containing IEEE 754 float16 bits.
 * Used when a model's input dtype is float16.
 */
function toFloat16Bits(src: Float32Array): Uint16Array {
  const dst = new Uint16Array(src.length);
  const f32 = new Float32Array(1);
  const u32 = new Uint32Array(f32.buffer);
  for (let i = 0; i < src.length; i += 1) {
    f32[0] = src[i] ?? 0;
    const bits = u32[0] ?? 0;
    const sign = (bits >>> 31) & 0x1;
    const exp = (bits >>> 23) & 0xff;
    const frac = bits & 0x7fffff;
    let f16exp: number;
    let f16frac: number;
    if (exp === 0) {
      f16exp = 0;
      f16frac = frac >>> 13;
    } else if (exp === 0xff) {
      f16exp = 0x1f;
      f16frac = frac !== 0 ? 0x200 : 0;
    } else {
      const e = exp - 127 + 15;
      if (e <= 0) {
        f16exp = 0;
        f16frac = e < -10 ? 0 : (frac | 0x800000) >>> (14 - e);
      } else if (e >= 31) {
        f16exp = 0x1f;
        f16frac = 0;
      } else {
        f16exp = e;
        f16frac = frac >>> 13;
      }
    }
    dst[i] = (sign << 15) | (f16exp << 10) | f16frac;
  }
  return dst;
}

/** Bilinear resize of a flat NHWC Float32Array. */
function bilinearResizeNhwc(
  src: Float32Array,
  srcW: number,
  srcH: number,
  dstW: number,
  dstH: number,
  channels: number,
): Float32Array {
  const dst = new Float32Array(dstW * dstH * channels);
  const scaleX = srcW / dstW;
  const scaleY = srcH / dstH;
  for (let y = 0; y < dstH; y += 1) {
    for (let x = 0; x < dstW; x += 1) {
      const sx = (x + 0.5) * scaleX - 0.5;
      const sy = (y + 0.5) * scaleY - 0.5;
      const x0 = Math.max(0, Math.floor(sx));
      const y0 = Math.max(0, Math.floor(sy));
      const x1 = Math.min(srcW - 1, x0 + 1);
      const y1 = Math.min(srcH - 1, y0 + 1);
      const wx = sx - Math.floor(sx);
      const wy = sy - Math.floor(sy);
      const dstBase = (y * dstW + x) * channels;
      for (let c = 0; c < channels; c += 1) {
        const v00 = src[(y0 * srcW + x0) * channels + c];
        const v10 = src[(y0 * srcW + x1) * channels + c];
        const v01 = src[(y1 * srcW + x0) * channels + c];
        const v11 = src[(y1 * srcW + x1) * channels + c];
        dst[dstBase + c] =
          v00 * (1 - wx) * (1 - wy) + v10 * wx * (1 - wy) + v01 * (1 - wx) * wy + v11 * wx * wy;
      }
    }
  }
  return dst;
}

/** Nearest-neighbour resize of a flat 1-channel Uint8Array. */
function nnResizeMask(
  src: Uint8Array,
  srcW: number,
  srcH: number,
  dstW: number,
  dstH: number,
): Uint8Array {
  const dst = new Uint8Array(dstW * dstH);
  const scaleX = srcW / dstW;
  const scaleY = srcH / dstH;
  for (let y = 0; y < dstH; y += 1) {
    const sy = Math.min(srcH - 1, Math.floor(y * scaleY));
    for (let x = 0; x < dstW; x += 1) {
      const sx = Math.min(srcW - 1, Math.floor(x * scaleX));
      dst[y * dstW + x] = src[sy * srcW + sx];
    }
  }
  return dst;
}

/** Per-pixel sigmoid, result written into `out`. */
function sigmoidInPlace(data: Float32Array, out: Float32Array): void {
  for (let i = 0; i < data.length; i += 1) {
    const v = data[i];
    out[i] = 1 / (1 + Math.exp(-Math.max(-60, Math.min(60, v))));
  }
}

/** IoU between two xyxy boxes. */
function iouXyxy(a: [number, number, number, number], b: [number, number, number, number]): number {
  const ix1 = Math.max(a[0], b[0]);
  const iy1 = Math.max(a[1], b[1]);
  const ix2 = Math.min(a[2], b[2]);
  const iy2 = Math.min(a[3], b[3]);
  const iw = Math.max(0, ix2 - ix1);
  const ih = Math.max(0, iy2 - iy1);
  const inter = iw * ih;
  if (inter === 0) return 0;
  const aArea = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]);
  const bArea = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
  const union = aArea + bArea - inter;
  return union > 0 ? inter / union : 0;
}

type YoloDet = {
  box: [number, number, number, number]; // xyxy in 640-px space
  score: number;
  coeffs: Float32Array; // 32 mask coefficients
};

/** Greedy NMS on YOLO detections (sorted descending by score). */
function greedyNms(dets: YoloDet[], iouThreshold: number): YoloDet[] {
  dets.sort((a, b) => b.score - a.score);
  const keep: YoloDet[] = [];
  const suppressed = new Uint8Array(dets.length);
  for (let i = 0; i < dets.length; i += 1) {
    if (suppressed[i]) continue;
    keep.push(dets[i]);
    for (let j = i + 1; j < dets.length; j += 1) {
      if (!suppressed[j] && iouXyxy(dets[i].box, dets[j].box) > iouThreshold) {
        suppressed[j] = 1;
      }
    }
  }
  return keep;
}

/**
 * Full YOLO-seg inference pipeline for a single patch.
 *
 * Inputs
 *   - session: loaded ONNX InferenceSession
 *   - rawData: flat Float32Array in NHWC order (H × W × 3)
 *   - patchW / patchH: original patch dimensions
 *   - normalization: divide pixel values by this to get 0–1 range
 *   - scoreThreshold / iouThreshold: detection filters
 *
 * Output layout (Ultralytics ONNX, static 640 × 640, NMS off):
 *   output0: [1, 37, 8400]  — 4 bbox + 1 class + 32 proto coefficients per anchor
 *   output1: [1, 32, 160, 160] — prototype masks
 */
async function runYoloSegPatchInference(
  session: InferenceSession,
  rawData: Float32Array,
  patchW: number,
  patchH: number,
  normalization: number,
  scoreThreshold: number,
  iouThreshold: number,
  precision: "fp16" | "fp32" = "fp32",
): Promise<OnnxPatchResponse> {
  const ort = await getOrtModule();
  const norm = normalization > 0 ? 1 / normalization : 1 / 255;

  // 1. Normalise → NHWC float [patchH, patchW, 3]
  const numSrcPixels = patchW * patchH * 3;
  const normalised = new Float32Array(numSrcPixels);
  for (let i = 0; i < numSrcPixels; i += 1) {
    normalised[i] = Math.min(Math.max((rawData[i] ?? 0) * norm, 0), 1);
  }

  // 2. Resize NHWC → [640, 640, 3]
  const resized = bilinearResizeNhwc(
    normalised,
    patchW,
    patchH,
    YOLO_INPUT_SIZE,
    YOLO_INPUT_SIZE,
    3,
  );

  // 3. NHWC → NCHW → tensor [1, 3, 640, 640]
  const nchw = new Float32Array(3 * YOLO_INPUT_SIZE * YOLO_INPUT_SIZE);
  for (let c = 0; c < 3; c += 1) {
    for (let y = 0; y < YOLO_INPUT_SIZE; y += 1) {
      for (let x = 0; x < YOLO_INPUT_SIZE; x += 1) {
        nchw[c * YOLO_INPUT_SIZE * YOLO_INPUT_SIZE + y * YOLO_INPUT_SIZE + x] =
          resized[(y * YOLO_INPUT_SIZE + x) * 3 + c];
      }
    }
  }

  const inputName = session.inputNames[0] ?? "images";
  const tensorData = precision === "fp16" ? toFloat16Bits(nchw) : nchw;
  const tensorType = precision === "fp16" ? "float16" : "float32";
  const feeds: Record<string, InstanceType<typeof ort.Tensor>> = {
    [inputName]: new ort.Tensor(tensorType, tensorData, [1, 3, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE]),
  };
  const outputs = await session.run(feeds);

  // 4. Parse outputs
  const out0Name = session.outputNames[0] ?? "output0";
  const out1Name = session.outputNames[1] ?? "output1";
  const out0 = outputs[out0Name];
  const out1 = outputs[out1Name];

  if (!out0 || !out1) {
    throw new Error(`YOLO model missing expected outputs. Got: ${Object.keys(outputs).join(", ")}`);
  }

  // out0: [1, 37, numAnchors]
  const [, featDim, numAnchors] = out0.dims as [number, number, number];
  const numClasses = featDim - 4 - YOLO_NUM_PROTOS; // = 1
  const det0Data = out0.data as Float32Array;

  // out1: [1, 32, protoH, protoW]
  const [, , protoH, protoW] = out1.dims as [number, number, number, number];
  const protoData = out1.data as Float32Array;
  const protoPixels = protoH * protoW;

  // 5. Decode detections: out0 is [1, 37, 8400] → iterate over anchors
  const dets: YoloDet[] = [];
  for (let a = 0; a < numAnchors; a += 1) {
    // scores: Ultralytics already applies sigmoid in the graph for class scores
    let maxScore = 0;
    for (let cls = 0; cls < numClasses; cls += 1) {
      const s = det0Data[(4 + cls) * numAnchors + a] as number;
      if (s > maxScore) maxScore = s;
    }
    if (maxScore < scoreThreshold) continue;

    const cx = det0Data[0 * numAnchors + a] as number;
    const cy = det0Data[1 * numAnchors + a] as number;
    const w = det0Data[2 * numAnchors + a] as number;
    const h = det0Data[3 * numAnchors + a] as number;
    const x1 = Math.max(0, cx - w / 2);
    const y1 = Math.max(0, cy - h / 2);
    const x2 = Math.min(YOLO_INPUT_SIZE, cx + w / 2);
    const y2 = Math.min(YOLO_INPUT_SIZE, cy + h / 2);

    if (x2 <= x1 || y2 <= y1) continue;

    const coeffs = new Float32Array(YOLO_NUM_PROTOS);
    for (let k = 0; k < YOLO_NUM_PROTOS; k += 1) {
      coeffs[k] = det0Data[(4 + numClasses + k) * numAnchors + a] as number;
    }

    dets.push({ box: [x1, y1, x2, y2], score: maxScore, coeffs });
  }

  // 6. NMS
  const kept = greedyNms(dets, iouThreshold);

  // 7. Build combined binary semantic mask at actual proto resolution
  const semanticMask = new Uint8Array(protoPixels); // 0 = background, 1 = field
  const rawMask = new Float32Array(protoPixels);
  const sigMask = new Float32Array(protoPixels);

  // Scale from model input pixel space → proto pixel space
  const protoScaleX = protoW / YOLO_INPUT_SIZE;
  const protoScaleY = protoH / YOLO_INPUT_SIZE;

  for (const det of kept) {
    // Linear combination: coeffs @ protos → [protoPixels]
    rawMask.fill(0);
    for (let k = 0; k < YOLO_NUM_PROTOS; k += 1) {
      const coeff = det.coeffs[k] ?? 0;
      const protoOffset = k * protoPixels;
      for (let px = 0; px < protoPixels; px += 1) {
        rawMask[px] += coeff * (protoData[protoOffset + px] ?? 0);
      }
    }
    sigmoidInPlace(rawMask, sigMask);

    // Crop to detection bbox (scaled to proto space) and threshold
    const bx1 = Math.max(0, Math.floor(det.box[0] * protoScaleX));
    const by1 = Math.max(0, Math.floor(det.box[1] * protoScaleY));
    const bx2 = Math.min(protoW, Math.ceil(det.box[2] * protoScaleX));
    const by2 = Math.min(protoH, Math.ceil(det.box[3] * protoScaleY));

    for (let py = by1; py < by2; py += 1) {
      for (let px = bx1; px < bx2; px += 1) {
        const idx = py * protoW + px;
        if ((sigMask[idx] ?? 0) >= 0.5) {
          semanticMask[idx] = 1;
        }
      }
    }
  }

  // 8. Nearest-neighbour resize mask to original patch size
  const finalMask = nnResizeMask(semanticMask, protoW, protoH, patchW, patchH);

  return {
    mask: Array.from(finalMask),
    width: patchW,
    height: patchH,
    classes: ["background", "field"],
    confidences: [],
  };
}

export const runPatchInferenceClient = async (
  session: InferenceSession,
  payload: OnnxPatchRequest,
): Promise<OnnxPatchResponse> => {
  const option = getModelOption(payload.modelId);
  const floatData = new Float32Array(payload.patchData);

  if (option.modelType === "yolo-seg") {
    return runYoloSegPatchInference(
      session,
      floatData,
      payload.patchWidth,
      payload.patchHeight,
      payload.normalization,
      payload.scoreThreshold,
      0.3,
      option.precision ?? "fp32",
    );
  }

  // FTW UNet path: upsample 2× then run semantic segmentation
  const {
    data: upsampledData,
    width: upsampledWidth,
    height: upsampledHeight,
  } = upsamplePatch(floatData, payload.patchWidth, payload.patchHeight, 2);

  const result = await runSegmentationInference(
    session,
    {
      data: upsampledData,
      width: upsampledWidth,
      height: upsampledHeight,
      normalization: payload.normalization,
    },
    option.classes ?? [],
    payload.scoreThreshold,
  );

  return {
    mask: result.mask,
    width: result.width,
    height: result.height,
    classes: result.classes,
    confidences: result.confidences,
  };
};
