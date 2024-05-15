//@ts-ignore
import init, { greet, reverseSearch} from 'rs_crate';
// Don't worry if vscode told you can't find my-crate
// It's because you're using a local crate
// after yarn dev, wasm-pack plugin will install my-crate for you
import { mnistConsts, loadMnist, MNIST } from './data-loader';
import { signal, effect } from "@preact/signals-core";
import * as tfc from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl'

const squareSize = 20;
const numImages = squareSize * squareSize;
const imagesIndices = signal([...Array(numImages).keys()]);
const imgSize = mnistConsts.numRows * mnistConsts.numColumns;
const projectionDim = 2;
const numClasses = 10;
const featureVectorSize = imgSize * numClasses;

async function drawMinist() {
  console.log("Drawing")

  const totalWidth = squareSize * mnistConsts.numColumns;
  const totalHeight = squareSize * mnistConsts.numRows;
  const canvas = document.querySelector("canvas") as HTMLCanvasElement;
  canvas.height = totalHeight;
  canvas.width = totalWidth;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    console.log("Return from draw");
    return;
  }
  ctx.font = "bold 10px serif";
  ctx.fillStyle = "green";
  ctx.textBaseline = "top";
  const canvasData = ctx.getImageData(0, 0, totalWidth, totalHeight);
  const mnist = await loadMnist();
  const trainingSet = mnist.train.set;
  const trainLabels = mnist.train.labels;
  interface Label { value: number, x: number, y: number, imgNum: number }
  console.log("loaded mnist");

  effect(() => {
    const indices = imagesIndices.value;
    console.log(indices, numImages);
    const labels: Label[] = [];

    for (let i = 0; i < numImages; ++i) {
      const imageIndex = indices[i];
      const image = trainingSet[imageIndex];
      const row = Math.trunc(i / squareSize);
      const rowOffset = row * mnistConsts.numColumns * mnistConsts.numRows * 4 * squareSize;
      const col = i % squareSize;
      const xoffset = col * mnistConsts.numColumns * 4;
      const yoffset = totalWidth * 4;
      for (let y = 0; y < mnistConsts.numRows; ++y) {
        for (let x = 0; x < mnistConsts.numColumns; ++x) {
          const mnistIndex = x + y * mnistConsts.numColumns;
          const pixel = image[mnistIndex];
          const canvasIndex = (rowOffset) + (xoffset + (x * 4)) + (yoffset * y);
          for (let c = 0; c < 3; ++c) {
            canvasData.data[canvasIndex + c] = pixel;
          }
          canvasData.data[canvasIndex + 3] = 255;
        }
      }
      labels.push({ value: trainLabels[imageIndex], x: mnistConsts.numColumns * col, y: mnistConsts.numRows * row, imgNum: imageIndex });
    }
    ctx.putImageData(canvasData, 0, 0);
    for (let label of labels) {
      ctx.fillText(label.value.toString(), label.x, label.y);
    }
  })
};

window.addEventListener("load", (event) => {
  drawMinist();

  const button = document.querySelector("#resample");
  console.log(button);
  button?.addEventListener("click", resampleClick);


});


init().then(async () => {
  console.log('init wasm-pack');
  const initialised = await tfc.setBackend('webgl');
  console.log("Initialised", initialised);
  const mnist = await loadMnist();
  reverseSearchLoop(mnist.train.set, mnist.train.labels);
  //greet('from vite!');
}).then();

export function resampleClick(ev: Event) {
  const samples: number[] = new Array(numImages);
  for (let i = 0; i < numImages; ++i) {
    samples[i] = Math.floor(Math.random() * mnistConsts.numImages);
  }
  imagesIndices.value = samples;
}



function writeResult(param: Float64Array, decomposition: Uint8Array){
  console.log("param", param);
  console.log("decomposition", decomposition);
}

function reverseSearchLoop(trainingDataBytes: Uint8Array[], trainingLabels: Uint8Array) {

  // numImages x imgSize
  const trainingData = tfc.tensor(trainingDataBytes);
  console.log(trainingData);

  // 1 x imgSize
  let param = tfc.ones([featureVectorSize]);
  const numIterations = 1;
  for (let iter = 0; iter < numIterations; ++iter) {
    // featureVectorSize x D
    const projected = tfc.tidy(() => {
      const directions: tfc.Tensor<tfc.Rank.R2> = tfc.randomStandardNormal([featureVectorSize, projectionDim])
      const projection = tfc.concat2d([directions, tfc.reshape(param, [-1, 1])], 1);

      // Because the feature vector is sparse, we only need
      // to use a subset of the projection vector.
      // imgSize x numClasses x D + 1
      console.time("projection");
      const reshaped = tfc.reshape(projection, [numClasses, imgSize, -1]);
      const projected = tfc.matMul(trainingData, reshaped);
      console.timeEnd("projection")
      //const synced = projected.dataSync();

      return projected;
    });
    const rsInput = projected.dataSync() as Float32Array;
    reverseSearch(rsInput, numImages, numClasses, projectionDim + 1, writeResult);

  }




}