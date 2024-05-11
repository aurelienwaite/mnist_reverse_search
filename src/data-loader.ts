import { Matrix } from 'ml-matrix';

interface DataSet {
  dataset: string;
  value: Uint8Array[]
}

export interface MNIST{
  train: {
    set: Uint8Array[],
    labels: Uint8Array
  }
  test: {
    set: Uint8Array[],
    labels: Uint8Array,
  }
}

type DataSetName = "train" | "test";
type Images = "images";
type Labels = "labels";
type DataSetType = Images | Labels;

const objectStoreName = "dataset";

export const mnistConsts = {
  trainDataMagicNum: 2051,
  trainLabelMagicNum: 2049,
  numImages: 60000,
  numRows: 28,
  numColumns: 28,
}

function assertion(offset: number, test: number, dv: DataView){ if (!dv.getUint32(offset * 4)) throw new Error(`assertion failed ${offset}`) }

function makeKey(set: DataSetName, data: DataSetType): string {
  return `${set}_${data}`
}

async function getDb(db_name: string = "MNIST"): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(db_name);
    request.onerror = (event) => {
      reject("Cannot get access to db");
    };
    request.onupgradeneeded = (event) => {
      const target = event.target as IDBOpenDBRequest;
      const db = target.result;
      db.createObjectStore(objectStoreName, { keyPath: "dataset" });
    };
    request.onsuccess = (event) => {
      if (event.target) {
        const target = event.target as IDBOpenDBRequest;
        const db = target.result;
        resolve(db);
      }
    };
  })
};


async function getDataset(set: DataSetName, setType: Images): Promise<Uint8Array[] | undefined>;
async function getDataset(set: DataSetName, setType: Labels): Promise<Uint8Array | undefined>;

async function getDataset(set: DataSetName, setType: DataSetType): Promise<Uint8Array[] | Uint8Array | undefined>{
  return new Promise(async (resolve, reject) => {
    const db = await getDb();
    const transaction = db.transaction([objectStoreName], "readonly");
    transaction.onerror = (ev) => {
      reject("Transaction failed");
    }
    const os = transaction.objectStore(objectStoreName);
    const objectStoreRequest = os.get(makeKey(set, setType));
    let value: DataSet | undefined = undefined;
    objectStoreRequest.onsuccess = (ev) => {
      value = objectStoreRequest.result as DataSet | undefined;
    }
    objectStoreRequest.onerror = (ev) => {
      reject("obs error");
    }
    transaction.oncomplete = (ev) => {
      if (value){
        if (setType === "images"){
          // Indexed db strips the class type away from the object. We have
          // to do this nastiness to get it back
          //const matrixObject: any = value.value;
          //const proto = Matrix.zeros(0,0);
          //Object.setPrototypeOf(matrixObject, proto); 
          resolve(value.value);
        }else{
          resolve(value.value);
        }

      }
      resolve(undefined);
    }
  });
}

async function addDataset(set: DataSetName, data: DataSetType, value: Uint8Array[] | Uint8Array): Promise<undefined> {
  return new Promise(async (resolve, reject) => {
    const db = await getDb();
    const transaction = db.transaction([objectStoreName], "readwrite");
    transaction.onerror = (ev) => {
      reject("Transaction failed");
    }
    const os = transaction.objectStore(objectStoreName);
    const objectStoreRequest = os.add({
      dataset: makeKey(set, data),
      value: value
    });
    objectStoreRequest.onerror = (ev) => {
      if (ev.target){
        console.log((ev.target as IDBRequest).error?.message)
      }
      reject("obs error");
    }
    transaction.oncomplete = (ev) => {
      resolve(undefined);
    }
  });
}

async function loadMnistTrainLabels(): Promise<Uint8Array>{
  const dsName = "train";
  const dsType = "labels";
  let dbRes = await getDataset(dsName, dsType);
  if (dbRes) {
    console.log("Found indexed db result for labels")
    return dbRes;
  }
  const response = await fetch("./data/train-labels.idx1-ubyte");
  if (!response.body){
    throw new Error("Label body not found");
  }
  const buffer = await response.arrayBuffer();
  const dv = new DataView(buffer);
  assertion(0, mnistConsts.trainLabelMagicNum, dv);
  assertion(1, mnistConsts.numImages, dv);
  const labelArray = new Uint8Array(mnistConsts.numImages);
  for (let i=0; i<mnistConsts.numImages;++i){
    labelArray[i] = dv.getUint8(4 * 2 + i);
  }
  await addDataset(dsName, dsType, labelArray)
  return labelArray;
}

async function loadMnistTrainSet(): Promise<Uint8Array[]>{
  const dsName = "train";
  const dsType = "images";
  let dbRes = await getDataset(dsName, dsType);
  if (dbRes) {
    console.log("Found indexed db result")
    return dbRes;
  }
  const dataMatrix: Uint8Array[] = new Array(mnistConsts.numImages);
  await fetch("./data/train-images.idx3-ubyte")
    // Retrieve its body as ReadableStream
    .then(async (response) => {
      if (!response.body) {
        throw new Error("Body not found!");
      }
      let lastImageRead = 0;
      let remainingBytes = 0;

      const reader = response.body.getReader();
      await reader.read().then(async function loadChunk({ done, value }) {
        if (!value) {
          return;
        }
        console.time('loadChunk');
        const dv = new DataView(value.buffer);
        const initialised = lastImageRead > 0;
        if (!initialised) {
          assertion(0, mnistConsts.trainDataMagicNum, dv);
          assertion(1, mnistConsts.numImages, dv);
          assertion(2, mnistConsts.numRows, dv);
          assertion(3, mnistConsts.numColumns, dv);
        }
        const imgSize = mnistConsts.numColumns * mnistConsts.numRows;
        const initialisationOffset = initialised ? 0 : 4 * 4;

        const fillTruncated = (globalImageNum: number) => {
          console.log(`Filling remaining bytes ${remainingBytes} for image ${globalImageNum}`)
          const matrixOffset = imgSize - remainingBytes;
          for (let j = 0; j < remainingBytes; ++j) {
            dataMatrix[globalImageNum][matrixOffset + j] = dv.getUint8(j);
          }          
        }
        
        const fillData = (chunkImageNum: number, globalImageNum: number): number => {
          const start = chunkImageNum * imgSize + initialisationOffset + remainingBytes;
          const end = Math.min(start + imgSize, value.length)
          dataMatrix[globalImageNum] = new Uint8Array(imgSize);
          for (let j = 0; j < end - start; ++j) {
            dataMatrix[globalImageNum][j] = dv.getUint8(start + j);
          }
          return (start + imgSize) - value.length;
        }

        if (remainingBytes > 0) {
          fillTruncated(lastImageRead - 1);
        }
        const numBytes = value.length - remainingBytes - initialisationOffset;
        const numImages = Math.trunc(numBytes / imgSize) + 1;
        let lastImageRemainingBytes = 0;
        for (let i = 0; i < numImages; ++i) {
          lastImageRemainingBytes = fillData(i, lastImageRead++);
        }
        remainingBytes = lastImageRemainingBytes;
        console.log(`Loaded ${numImages}, total images loaded ${lastImageRead}`)
        console.timeEnd("loadChunk");
        if (done) {
          return;
        } else {
          await reader.read().then(loadChunk);
        }
      })
      return dataMatrix;
    }).then((dataMatrix) => {
      addDataset(dsName, dsType, dataMatrix)
    })
  
  return dataMatrix;
}

export async function loadMnist(): Promise<MNIST>{
  const [trainLabels, trainSet] = await Promise.all([loadMnistTrainLabels(), loadMnistTrainSet()]);
  return {
    train: {
      set: trainSet,
      labels: trainLabels,
    },
    test: {
      set: [],
      labels: new Uint8Array(),
    }
  }
}