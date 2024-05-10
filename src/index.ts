//@ts-ignore
import init, { greet } from 'my-crate';
// Don't worry if vscode told you can't find my-crate
// It's because you're using a local crate
// after yarn dev, wasm-pack plugin will install my-crate for you
import { mnistConsts, loadMnist, MNIST } from './data-loader';


window.addEventListener("load", (event) => {
  console.log("Drawing")
  const squareSize = 20;
  const totalWidth = squareSize * mnistConsts.numColumns;
  const totalHeight = squareSize * mnistConsts.numRows;
  const canvas = document.querySelector("canvas") as HTMLCanvasElement;
  canvas.height=totalHeight;
  canvas.width=totalWidth;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    console.log("Return from draw");
    return;
  }
  ctx.font = "bold 10px serif";
  ctx.fillStyle = "green";
  ctx.textBaseline = "top";
  const canvasData = ctx.getImageData(0, 0, totalWidth, totalHeight);
  loadMnist().then((mnist) => {
    const trainingSet = mnist.train.set;
    const trainLabels = mnist.train.labels;
    const numImages = squareSize * squareSize;
    
    interface Label {value: number, x: number, y: number, imgNum: number }
    const labels: Label[] = [];

    for (let i = 0; i < numImages; ++i) {
      const imageIndex = Math.floor(Math.random() * mnistConsts.numImages)
      const image = trainingSet.getRow(imageIndex);
      const row = Math.trunc(i / squareSize);
      const rowOffset = row * mnistConsts.numColumns * mnistConsts.numRows * 4 * squareSize;
      const col = i % squareSize;
      const xoffset = col * mnistConsts.numColumns * 4;
      const yoffset = totalWidth * 4;
      for (let y = 0; y < mnistConsts.numRows; ++y) {
        for (let x = 0; x < mnistConsts.numColumns; ++x) {
          const mnistIndex = x + y * mnistConsts.numColumns;
          const pixel = image[mnistIndex];
          const canvasIndex = (rowOffset) + (xoffset + (x*4)) + (yoffset * y);
          for (let c = 0; c < 3; ++c){
            canvasData.data[canvasIndex + c] = pixel;
          }
          canvasData.data[canvasIndex + 3] = 255;
        }
      }
      labels.push({value: trainLabels[imageIndex], x: mnistConsts.numColumns * col, y: mnistConsts.numRows * row, imgNum: imageIndex});
    }
    ctx.putImageData(canvasData, 0, 0);
    for (let label of labels){
      ctx.fillText(label.value.toString(), label.x, label.y);
    }
    
  });
});

init().then(() => {
  console.log('init wasm-pack');
  //greet('from vite!');
});