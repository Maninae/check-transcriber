/**
 * The opt-in handwriting reader ("Read handwriting" in settings): TrOCR-small-handwritten
 * as exported by Xenova for transformers.js (fp32 encoder + int8 merged past-KV decoder,
 * the combination experiments/field_reading/export/BROWSER_NOTE.md recommends: full int8
 * breaks the encoder). Run on the worker's own onnxruntime-web instead of through
 * transformers.js, which would bring a second (dev-build) runtime; the decode loop is a
 * port of learned/trocr_onnx_decoding.py `greedy_decode_merged_past`.
 *
 * - Download: three files from the Hugging Face Hub, pinned to one revision (cdn_config.js);
 *   the service worker caches them after the first fetch, so it happens once.
 * - Input: grey + autocontrast(2%) + bicubic 384 x 384 (Pillow-exact, pillow_image_operations.js),
 *   (x / 255 - 0.5) / 0.5, grey copied to three channels.
 * - Output: text + confidence = exp(mean log-prob of the chosen tokens, EOS included).
 * The model weights are third-party (Microsoft, fine-tuned on IAM; see the README licence note).
 */

import { autocontrastLuma, convertRgbToLuma, resizeLumaBicubic } from "./pillow_image_operations.js";

const INPUT_SIZE = 384;
const AUTOCONTRAST_CUTOFF_PERCENT = 2;
const PIXEL_VALUE_SCALE = 255;
const MAXIMUM_NEW_TOKENS = 48; // trocr_reader.py TROCR_MAX_NEW_TOKENS
const END_OF_SEQUENCE_ID = 2; // decoding starts from </s> and stops at it
const SPECIAL_TOKEN_IDS = new Set([0, 1, 2, 3]); // <s> <pad> </s> <unk>
const MASK_TOKEN_ID = 64001; // ids at or above it are not text pieces
const DECODER_ATTENTION_HEADS = 8;
const DECODER_HEAD_SIZE = 32;
const PAST_INPUT_PREFIX = "past_key_values.";
const PRESENT_OUTPUT_PREFIX = "present.";
const USE_CACHE_BRANCH_INPUT = "use_cache_branch";
const SENTENCEPIECE_SPACE = "▁";
const BYTES_PER_MEGABYTE = 1e6;

/** Fetches a URL into a Uint8Array, reporting cumulative bytes through `onBytes(received)`. */
function fetchBytesWithProgress(url, onBytes) {
  return fetch(url).then((response) => {
    if (!response.ok) throw new Error(`download failed (${response.status}) for ${url}`);
    if (!response.body) return response.arrayBuffer().then((buffer) => new Uint8Array(buffer));
    const reader = response.body.getReader();
    const chunks = [];
    let received = 0;
    const pump = () => reader.read().then(({ done, value }) => {
      if (done) {
        const bytes = new Uint8Array(received);
        let offset = 0;
        for (const chunk of chunks) {
          bytes.set(chunk, offset);
          offset += chunk.length;
        }
        return bytes;
      }
      chunks.push(value);
      received += value.length;
      onBytes(received);
      return pump();
    });
    return pump();
  });
}

/**
 * Downloads (or reads from the service-worker cache) and opens the reader.
 * `urls`: `{ encoder, decoder, tokenizer, totalMegabytes }`; `onProgress(text)`.
 * Resolves to a reader object for `readHandwrittenCrop`.
 */
export function loadHandwritingReader(ort, urls, onProgress) {
  const receivedByFile = { encoder: 0, decoder: 0, tokenizer: 0 };
  const report = () => {
    const receivedMegabytes = Object.values(receivedByFile).reduce((sum, bytes) => sum + bytes, 0) / BYTES_PER_MEGABYTE;
    onProgress(`Downloading the handwriting reader: ${Math.round(receivedMegabytes)} of ${urls.totalMegabytes} MB`);
  };
  const track = (fileKey) => (received) => { receivedByFile[fileKey] = received; report(); };
  report();
  return Promise.all([
    fetchBytesWithProgress(urls.encoder, track("encoder")),
    fetchBytesWithProgress(urls.decoder, track("decoder")),
    fetchBytesWithProgress(urls.tokenizer, track("tokenizer")),
  ]).then(([encoderBytes, decoderBytes, tokenizerBytes]) => {
    onProgress("Preparing the handwriting reader");
    const vocabulary = JSON.parse(new TextDecoder().decode(tokenizerBytes)).model.vocab.map(([piece]) => piece);
    // Separate options objects: onnxruntime-web writes into them.
    return Promise.all([
      ort.InferenceSession.create(encoderBytes, { executionProviders: ["wasm"] }),
      ort.InferenceSession.create(decoderBytes, { executionProviders: ["wasm"] }),
    ]).then(([encoderSession, decoderSession]) => ({ encoderSession, decoderSession, vocabulary }));
  });
}

/** The encoder input for one RGBA field crop: Float32Array (1, 3, 384, 384). */
export function buildHandwritingPixelValues(rgbaPixels, width, height) {
  const luma = convertRgbToLuma(rgbaPixels, width, height, 4);
  const resized = resizeLumaBicubic(autocontrastLuma(luma, AUTOCONTRAST_CUTOFF_PERCENT), width, height, INPUT_SIZE, INPUT_SIZE);
  const planeSize = INPUT_SIZE * INPUT_SIZE;
  const pixelValues = new Float32Array(3 * planeSize);
  for (let index = 0; index < planeSize; index += 1) {
    const normalized = (Math.fround(resized[index] / PIXEL_VALUE_SCALE) - 0.5) / 0.5;
    pixelValues[index] = normalized;
    pixelValues[planeSize + index] = normalized;
    pixelValues[2 * planeSize + index] = normalized;
  }
  return pixelValues;
}

/** Token ids -> text, dropping specials (the Python codec's `decode`). */
export function decodeTokenIds(vocabulary, tokenIds) {
  const pieces = tokenIds.filter((id) => !SPECIAL_TOKEN_IDS.has(id) && id < MASK_TOKEN_ID && id < vocabulary.length)
    .map((id) => vocabulary[id]);
  return pieces.join("").replaceAll(SENTENCEPIECE_SPACE, " ").trim();
}

function logSoftmaxAt(logits, offset, length, index) {
  let maximum = -Infinity;
  for (let i = 0; i < length; i += 1) if (logits[offset + i] > maximum) maximum = logits[offset + i];
  let sum = 0;
  for (let i = 0; i < length; i += 1) sum += Math.exp(logits[offset + i] - maximum);
  return logits[offset + index] - maximum - Math.log(sum);
}

function emptyPastFeeds(ort, decoderSession) {
  const feeds = {};
  for (const name of decoderSession.inputNames) {
    if (name.startsWith(PAST_INPUT_PREFIX)) {
      feeds[name] = new ort.Tensor("float32", new Float32Array(0), [1, DECODER_ATTENTION_HEADS, 0, DECODER_HEAD_SIZE]);
    }
  }
  return feeds;
}

/** Resolves to `{ text, confidence }` for one RGBA field crop. */
export function readHandwrittenCrop(ort, reader, rgbaPixels, width, height) {
  const pixelValues = new ort.Tensor("float32", buildHandwritingPixelValues(rgbaPixels, width, height), [1, 3, INPUT_SIZE, INPUT_SIZE]);
  const encoderInputName = reader.encoderSession.inputNames[0];
  return reader.encoderSession.run({ [encoderInputName]: pixelValues }).then((encoderOutputs) => {
    const hiddenStates = encoderOutputs[reader.encoderSession.outputNames[0]];
    const tokens = [END_OF_SEQUENCE_ID];
    const chosenLogProbabilities = [];
    let past = emptyPastFeeds(ort, reader.decoderSession);
    const step = (stepIndex) => {
      const feeds = {
        input_ids: new ort.Tensor("int64", BigInt64Array.from([BigInt(tokens[tokens.length - 1])]), [1, 1]),
        encoder_hidden_states: hiddenStates,
        [USE_CACHE_BRANCH_INPUT]: new ort.Tensor("bool", Uint8Array.from([stepIndex > 0 ? 1 : 0]), [1]),
        ...past,
      };
      return reader.decoderSession.run(feeds).then((outputs) => {
        const logits = outputs.logits;
        const vocabularySize = logits.dims[2];
        const lastRowOffset = (logits.dims[1] - 1) * vocabularySize;
        let bestToken = 0;
        for (let token = 1; token < vocabularySize; token += 1) {
          if (logits.data[lastRowOffset + token] > logits.data[lastRowOffset + bestToken]) bestToken = token;
        }
        tokens.push(bestToken);
        chosenLogProbabilities.push(logSoftmaxAt(logits.data, lastRowOffset, vocabularySize, bestToken));
        const nextPast = { ...past };
        for (const [name, value] of Object.entries(outputs)) {
          // Cross-attention KV is computed at step 0; cache-branch steps return an empty placeholder.
          if (name.startsWith(PRESENT_OUTPUT_PREFIX) && value.size > 0) {
            nextPast[name.replace(PRESENT_OUTPUT_PREFIX, PAST_INPUT_PREFIX)] = value;
          }
        }
        past = nextPast;
        if (bestToken === END_OF_SEQUENCE_ID || stepIndex + 1 >= MAXIMUM_NEW_TOKENS) return null;
        return step(stepIndex + 1);
      });
    };
    return step(0).then(() => {
      const meanLogProbability = chosenLogProbabilities.reduce((sum, value) => sum + value, 0) / chosenLogProbabilities.length;
      return { text: decodeTokenIds(reader.vocabulary, tokens), confidence: Math.exp(meanLogProbability) };
    });
  });
}
