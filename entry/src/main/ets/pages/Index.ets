import { common } from '@kit.AbilityKit';
import { mindSporeLite } from '@kit.MindSporeLiteKit';
import { fileIo as fs } from '@kit.CoreFileKit';
import { image } from '@kit.ImageKit';
import { photoAccessHelper } from '@kit.MediaLibraryKit';
import { cameraPicker, camera } from '@kit.CameraKit';

interface BoundingBox {
  x: number;
  y: number;
  w: number;
  h: number;
  score: number;
  classId: number;
  className: string;
}

@Entry
@Component
struct Index {
  @State currentImageUri: string = '';
  @State isInferencing: boolean = false;
  @State isModelReady: boolean = false;

  @State displayWidth: number = 0;
  @State displayHeight: number = 0;
  @State detectionCount: number = 0;
  @State detectionLabels: string[] = [];

  private msModel: mindSporeLite.Model | undefined = undefined;
  private canvasContext: CanvasRenderingContext2D =
    new CanvasRenderingContext2D(new RenderingContextSettings(true));

  // NMS 可调参数
  private confidenceThreshold: number = 0.50;
  private iouThreshold: number = 0.40;

  private classNames: string[] = [
    "人", "自行车", "汽车", "摩托车", "飞机", "公交车", "火车", "卡车", "船", "红绿灯",
    "消防栓", "停止标志", "停车表", "长椅", "鸟", "猫", "狗", "马", "羊", "牛",
    "大象", "熊", "斑马", "长颈鹿", "背包", "雨伞", "手提包", "领带", "手提箱", "飞盘",
    "滑雪板", "单板滑雪", "运动球", "风筝", "棒球棒", "棒球手套", "滑板", "冲浪板", "网球拍", "瓶子",
    "红酒杯", "杯子", "叉子", "刀", "勺子", "碗", "香蕉", "苹果", "三明治", "橙子",
    "西兰花", "胡萝卜", "热狗", "比萨", "甜甜圈", "蛋糕", "椅子", "沙发", "盆栽", "床",
    "餐桌", "马桶", "电视", "笔记本电脑", "鼠标", "遥控器", "键盘", "手机", "微波炉", "烤箱",
    "烤面包机", "水槽", "冰箱", "书", "时钟", "花瓶", "剪刀", "泰迪熊", "吹风机", "牙刷"
  ];

  aboutToAppear() {
    this.initAIModel();
  }

  async initAIModel() {
    try {
      let context = this.getUIContext().getHostContext() as common.UIAbilityContext;
      let rawFile = await context.resourceManager.getRawFileContent('yolo26n.ms');

      let modelBuffer = rawFile.buffer.slice(
        rawFile.byteOffset,
        rawFile.byteOffset + rawFile.byteLength
      );

      let modelPath = context.filesDir + '/yolo26n.ms';
      let file = fs.openSync(modelPath, fs.OpenMode.READ_WRITE | fs.OpenMode.CREATE);
      fs.writeSync(file.fd, modelBuffer);
      fs.closeSync(file);

      let msContext: mindSporeLite.Context = { target: ['cpu'] };
      this.msModel = await mindSporeLite.loadModelFromFile(modelPath, msContext);

      if (!this.msModel) {
        this.isModelReady = false;
        return;
      }
      this.isModelReady = true;
      console.info('【YOLO Demo】AI 模型加载成功！');

    } catch (e) {
      console.error(`【YOLO Demo】initAIModel 失败: ${JSON.stringify(e)}`);
      this.isModelReady = false;
    }
  }

  private async pickFromGallery() {
    try {
      let photoPicker = new photoAccessHelper.PhotoViewPicker();
      let options = new photoAccessHelper.PhotoSelectOptions();
      options.MIMEType = photoAccessHelper.PhotoViewMIMETypes.IMAGE_TYPE;
      options.maxSelectNumber = 1;

      let result = await photoPicker.select(options);

      if (result && result.photoUris.length > 0) {
        this.currentImageUri = result.photoUris[0];
        await this.processImage(this.currentImageUri);
      }
    } catch (error) {
      console.error(`【YOLO Demo】图片选择失败: ${JSON.stringify(error)}`);
    }
  }

  private async takePhoto() {
    try {
      let context = this.getUIContext().getHostContext() as common.UIAbilityContext;
      let pickerProfile: cameraPicker.PickerProfile = {
        cameraPosition: camera.CameraPosition.CAMERA_POSITION_BACK
      };

      let pickerResult = await cameraPicker.pick(
        context,
        [cameraPicker.PickerMediaType.PHOTO],
        pickerProfile
      );

      if (pickerResult && pickerResult.resultUri) {
        this.currentImageUri = pickerResult.resultUri;
        await this.processImage(this.currentImageUri);
      }
    } catch (error) {
      console.error(`【YOLO Demo】拍摄照片失败: ${JSON.stringify(error)}`);
    }
  }

  private async processImage(uri: string) {
    if (!this.isModelReady) {
      console.error('【YOLO Demo】模型未就绪');
      return;
    }

    this.isInferencing = true;
    this.canvasContext.clearRect(0, 0, 4000, 4000);

    try {
      let file = fs.openSync(uri, fs.OpenMode.READ_ONLY);
      let imageSource = image.createImageSource(file.fd);
      let pixelMap = await imageSource.createPixelMap();
      fs.closeSync(file);

      await this.runModelInference(pixelMap);

      if (pixelMap) await pixelMap.release();
      if (imageSource) await imageSource.release();

    } catch (error) {
      console.error(`【YOLO Demo】图片处理或推理失败: ${JSON.stringify(error)}`);
    } finally {
      this.isInferencing = false;
    }
  }

  private async runModelInference(pixelMap: image.PixelMap) {
    let imgInfo = await pixelMap.getImageInfo();
    console.info(`【YOLO Demo】原始图片尺寸: ${imgInfo.size.width} x ${imgInfo.size.height}`);
    await pixelMap.scale(640 / imgInfo.size.width, 640 / imgInfo.size.height);

    let inputs = this.msModel!.getInputs();
    let inputTensor = inputs[0];

    console.info(`【YOLO Demo】模型输入张量: name="${inputTensor.name}", shape=[${inputTensor.shape}], ` +
      `dtype=${inputTensor.dtype}, format=${inputTensor.format}, elements=${inputTensor.elementNum}`);

    let scaledInfo = await pixelMap.getImageInfo();
    console.info(`【YOLO Demo】缩放后图片尺寸: ${scaledInfo.size.width} x ${scaledInfo.size.height}`);

    let pixelBytesBuffer = new ArrayBuffer(pixelMap.getPixelBytesNumber());
    await pixelMap.readPixelsToBuffer(pixelBytesBuffer);

    await new Promise<void>(resolve => setTimeout(resolve, 0));

    let rgbaArray = new Uint8Array(pixelBytesBuffer);
    let imageW = scaledInfo.size.width;
    let imageH = scaledInfo.size.height;
    let imageArea = imageW * imageH;
    let float32Array = new Float32Array(3 * imageArea);

    for (let i = 0; i < imageArea; i++) {
      let fi = i * 3;
      float32Array[fi + 0] = rgbaArray[i * 4] / 255.0;       // R
      float32Array[fi + 1] = rgbaArray[i * 4 + 1] / 255.0;   // G
      float32Array[fi + 2] = rgbaArray[i * 4 + 2] / 255.0;   // B
    }

    await new Promise<void>(resolve => setTimeout(resolve, 0));

    inputTensor.setData(float32Array.buffer);
    let outputs = (await this.msModel!.predict(inputs)) as mindSporeLite.MSTensor[];

    if (outputs && outputs.length > 0) {
      this.parseAndDrawBoxes(outputs);
    }
  }

  private calculateIoU(box1: BoundingBox, box2: BoundingBox): number {
    let xA = Math.max(box1.x, box2.x);
    let yA = Math.max(box1.y, box2.y);
    let xB = Math.min(box1.x + box1.w, box2.x + box2.w);
    let yB = Math.min(box1.y + box1.h, box2.y + box2.h);

    let interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
    if (interArea === 0) return 0;

    let box1Area = Math.max(0, box1.w * box1.h);
    let box2Area = Math.max(0, box2.w * box2.h);
    let unionArea = box1Area + box2Area - interArea;
    if (unionArea <= 0) return 0;

    return interArea / unionArea;
  }

  private nonMaxSuppressionByClass(
    boxes: BoundingBox[],
    iouThreshold: number
  ): BoundingBox[] {
    const classMap: Map<number, BoundingBox[]> = new Map();
    for (let i = 0; i < boxes.length; i++) {
      const cls: number = boxes[i].classId;
      if (!classMap.has(cls)) {
        classMap.set(cls, []);
      }
      let arr = classMap.get(cls);
      if (arr !== undefined) {
        arr.push(boxes[i]);
      }
    }

    const result: BoundingBox[] = [];
    const classEntries: [number, BoundingBox[]][] = Array.from(classMap.entries());
    for (let entryIdx = 0; entryIdx < classEntries.length; entryIdx++) {
      let classBoxes: BoundingBox[] = classEntries[entryIdx][1];
      classBoxes = classBoxes.slice().sort((a, b) => b.score - a.score);

      while (classBoxes.length > 0) {
        const best: BoundingBox = classBoxes[0];
        result.push(best);

        const remaining: BoundingBox[] = [];
        for (let i = 1; i < classBoxes.length; i++) {
          const iou: number = this.calculateIoU(best, classBoxes[i]);
          if (iou <= iouThreshold) {
            remaining.push(classBoxes[i]);
          }
        }
        classBoxes = remaining;
      }
    }

    return result;
  }

  private parseAndDrawBoxes(outputs: mindSporeLite.MSTensor[]) {
    let currentWidth = this.displayWidth;
    let currentHeight = this.displayHeight;

    this.canvasContext.clearRect(0, 0, 4000, 4000);
    if (!outputs || outputs.length === 0 || currentWidth === 0) return;

    let outTensor = outputs[0];
    let resultData = new Float32Array(outTensor.getData());
    let boxes: BoundingBox[] = [];

    let tensorShape: number[] = outTensor.shape;
    console.info(`【YOLO Demo】输出张量 shape: [${tensorShape}], 元素数: ${resultData.length}`);

    let numBoxes: number = 0;
    let paramsPerBox: number = 0;
    let isChannelFirst: boolean = false;

    if (tensorShape.length === 3) {
      let dim1: number = tensorShape[1];
      let dim2: number = tensorShape[2];
      if (dim1 === 6 && dim2 > 0) {
        numBoxes = dim2;
        paramsPerBox = dim1;
        isChannelFirst = true;
      } else if (dim2 === 6 && dim1 > 0) {
        numBoxes = dim1;
        paramsPerBox = dim2;
        isChannelFirst = false;
      }
    } else if (tensorShape.length === 2 && (resultData.length % 6 === 0)) {
      numBoxes = resultData.length / 6;
      paramsPerBox = 6;
      let midScore = resultData[Math.floor(resultData.length / 2)];
      if (midScore >= 0 && midScore <= 1.0) {
        isChannelFirst = false;
      } else {
        isChannelFirst = false;
      }
    }

    if (numBoxes === 0) {
      console.error(`【YOLO Demo】无法解析输出张量 shape: [${tensorShape}]`);
      return;
    }

    let maxVal = 0;
    let minVal = Infinity;
    for (let i = 0; i < resultData.length; i++) {
      if (resultData[i] > maxVal) maxVal = resultData[i];
      if (resultData[i] < minVal) minVal = resultData[i];
    }
    let isNormalized = maxVal <= 2.0 && minVal >= -0.5;

    console.info(
      `【YOLO Demo】格式: ${isChannelFirst ? 'channel-first' : 'row-major'}, ` +
        `盒子数: ${numBoxes}, 每盒参数: ${paramsPerBox}, ` +
        `坐标归一化: ${isNormalized} (范围 [${minVal.toFixed(2)}, ${maxVal.toFixed(2)}])`
    );

    let diagCount: number = Math.min(5, numBoxes);
    for (let i = 0; i < diagCount; i++) {
      let v: number[] = [];
      if (isChannelFirst) {
        v = [
          resultData[0 * numBoxes + i],
          resultData[1 * numBoxes + i],
          resultData[2 * numBoxes + i],
          resultData[3 * numBoxes + i],
          resultData[4 * numBoxes + i],
          resultData[5 * numBoxes + i]
        ];
      } else {
        let base = i * paramsPerBox;
        v = [
          resultData[base + 0], resultData[base + 1], resultData[base + 2],
          resultData[base + 3], resultData[base + 4], resultData[base + 5]
        ];
      }
      console.info(
        `【YOLO Demo】原始框[${i}]: [${v.map((x: number) => x.toFixed(4)).join(', ')}] ` +
          `→ 中心释义[x,y,w,h]中心=(${((v[0]+v[2])/2).toFixed(3)},${((v[1]+v[3])/2).toFixed(3)}) ` +
          `角点释义[x1,y1,x2,y2]=(${v[0].toFixed(3)},${v[1].toFixed(3)},${v[2].toFixed(3)},${v[3].toFixed(3)}) ` +
          `score=${v[4].toFixed(4)} classId_raw=${v[5].toFixed(3)}`
      );
    }

    for (let i = 0; i < numBoxes; i++) {
      let x1: number = 0;
      let y1: number = 0;
      let x2: number = 0;
      let y2: number = 0;
      let score: number = 0;
      let classId: number = 0;

      if (isChannelFirst) {
        x1 = resultData[0 * numBoxes + i];
        y1 = resultData[1 * numBoxes + i];
        x2 = resultData[2 * numBoxes + i];
        y2 = resultData[3 * numBoxes + i];
        score = resultData[4 * numBoxes + i];
        classId = resultData[5 * numBoxes + i];
      } else {
        let base = i * paramsPerBox;
        x1 = resultData[base + 0];
        y1 = resultData[base + 1];
        x2 = resultData[base + 2];
        y2 = resultData[base + 3];
        score = resultData[base + 4];
        classId = resultData[base + 5];
      }

      classId = Math.round(classId);

      if (score > this.confidenceThreshold) {
        let normX1 = isNormalized ? x1 : x1 / 640.0;
        let normY1 = isNormalized ? y1 : y1 / 640.0;
        let normX2 = isNormalized ? x2 : x2 / 640.0;
        let normY2 = isNormalized ? y2 : y2 / 640.0;

        let finalX = Math.max(0, normX1);
        let finalY = Math.max(0, normY1);
        let finalW = Math.max(0, normX2 - normX1);
        let finalH = Math.max(0, normY2 - normY1);

        if (finalX + finalW > 1.0) finalW = 1.0 - finalX;
        if (finalY + finalH > 1.0) finalH = 1.0 - finalY;

        boxes.push({
          x: finalX,
          y: finalY,
          w: finalW,
          h: finalH,
          score: score,
          classId: classId,
          className: this.classNames[classId] || `物品_${classId}`
        });
      }
    }

    let rawCount = boxes.length;
    let finalBoxes = this.nonMaxSuppressionByClass(boxes, this.iouThreshold);
    console.info(
      `【YOLO Demo】NMS 结果: ${rawCount} → ${finalBoxes.length} 个框 ` +
        `(conf=${this.confidenceThreshold}, iou=${this.iouThreshold})`
    );

    const classColors: string[] = [
      '#00FF00', '#FF4500', '#1E90FF', '#FFD700', '#FF1493',
      '#00CED1', '#FF8C00', '#9400D3', '#32CD32', '#FF6347',
      '#4682B4', '#FF69B4', '#7B68EE', '#00FA9A', '#DC143C',
      '#00BFFF', '#FFA500', '#8A2BE2', '#ADFF2F', '#FFB6C1'
    ];

    let labels: string[] = [];
    let classCount: Map<string, number> = new Map();
    for (let i = 0; i < finalBoxes.length; i++) {
      let name = finalBoxes[i].className;
      classCount.set(name, (classCount.get(name) || 0) + 1);
    }
    classCount.forEach((count: number, name: string) => {
      labels.push(`${name} ×${count}`);
    });
    this.detectionCount = finalBoxes.length;
    this.detectionLabels = labels;

    for (let i = 0; i < finalBoxes.length; i++) {
      let box = finalBoxes[i];

      let drawX = box.x * currentWidth;
      let drawY = box.y * currentHeight;
      let drawW = box.w * currentWidth;
      let drawH = box.h * currentHeight;

      let color = classColors[box.classId % classColors.length];

      this.canvasContext.strokeStyle = color;
      this.canvasContext.lineWidth = 5;
      this.canvasContext.strokeRect(drawX, drawY, drawW, drawH);

      const labelText: string = `${box.className} ${(box.score * 100).toFixed(0)}%`;
      this.canvasContext.font = '24px Arial';
      let textMetrics = this.canvasContext.measureText(labelText);
      let labelWidth: number = Math.max(textMetrics.width + 16, drawW);

      this.canvasContext.fillStyle = color;
      this.canvasContext.fillRect(drawX, drawY - 35, labelWidth, 35);

      this.canvasContext.fillStyle = '#000000';
      this.canvasContext.fillText(labelText, drawX + 8, drawY - 8);
    }
  }

  build() {
    Column() {
      // ── 顶部状态栏 ──
      Row() {
        Text('🎯 YOLO 目标检测')
          .fontSize(22)
          .fontWeight(FontWeight.Bolder)
          .fontColor('#FFFFFF')
          .letterSpacing(1)

        Blank()

        Row({ space: 6 }) {
          Circle({ width: 8, height: 8 })
            .fill(this.isModelReady ? '#4ADE80' : '#F87171')
            .shadow({ radius: 6, color: this.isModelReady ? '#4ADE80' : '#F87171' })
          Text(this.isModelReady ? '引擎就绪' : '加载中…')
            .fontSize(14)
            .fontWeight(FontWeight.Medium)
            .fontColor(this.isModelReady ? '#4ADE80' : '#F87171')
        }
        .padding({ left: 12, right: 12, top: 6, bottom: 6 })
        .backgroundColor('rgba(255,255,255,0.05)')
        .borderRadius(20)
      }
      .width('100%')
      .padding({ left: 24, right: 24, top: 16, bottom: 16 })
      .backgroundColor('#0F172A')

      // ── 图像与画布区 ──
      Scroll() {
        Column() {
          if (this.currentImageUri !== '') {
            Stack({ alignContent: Alignment.Bottom }) {
              Stack({ alignContent: Alignment.Center }) {
                Image(this.currentImageUri)
                  .width('100%')
                  .borderRadius(16)
                  .onAreaChange((oldArea: Area, newArea: Area) => {
                    this.displayWidth = newArea.width as number;
                    this.displayHeight = newArea.height as number;
                  })

                if (this.displayWidth > 0 && this.displayHeight > 0) {
                  Canvas(this.canvasContext)
                    .width(this.displayWidth)
                    .height(this.displayHeight)
                    .hitTestBehavior(HitTestMode.Transparent)
                }
              }
              .width('100%')
              .borderRadius(16)
              .shadow({
                radius: 20,
                color: 'rgba(0,0,0,0.5)',
                offsetX: 0,
                offsetY: 8
              })

              // ── 悬浮统计结果 ──
              if (this.detectionCount > 0 && !this.isInferencing) {
                Column({ space: 8 }) {
                  Row({ space: 8 }) {
                    Text(`📊 共 ${this.detectionCount} 个目标`)
                      .fontSize(14)
                      .fontWeight(FontWeight.Bold)
                      .fontColor('#FFFFFF')
                      .padding({ left: 12, right: 12, top: 6, bottom: 6 })
                      .backgroundColor('#3B82F6')
                      .borderRadius(12)
                  }

                  if (this.detectionLabels.length > 0) {
                    Flex({ wrap: FlexWrap.Wrap }) {
                      ForEach(this.detectionLabels.slice(0, 10), (label: string) => {
                        Text(label)
                          .fontSize(12)
                          .fontColor('#E2E8F0')
                          .padding({ left: 10, right: 10, top: 6, bottom: 6 })
                          .backgroundColor('rgba(255,255,255,0.15)')
                          .borderRadius(12)
                          .margin({ right: 6, bottom: 6 })
                      })
                    }
                  }
                }
                .width('90%')
                .padding(16)
                .margin({ bottom: 16 })
                .backgroundColor('rgba(15,23,42,0.85)')
                .backdropBlur(20)
                .borderRadius(20)
                .shadow({ radius: 20, color: 'rgba(0,0,0,0.6)', offsetY: 6 })
              }

              // ── 加载状态遮罩 ──
              if (this.isInferencing) {
                Column({ space: 16 }) {
                  LoadingProgress()
                    .width(48)
                    .height(48)
                    .color('#60A5FA')

                  Text('神经引擎解码中…')
                    .fontSize(16)
                    .fontColor('#F8FAFC')
                    // 🚀 修复点：已将 SemiBold 替换为通用的 Medium
                    .fontWeight(FontWeight.Medium)
                }
                .width('100%')
                .height('100%')
                .borderRadius(16)
                .backgroundColor('rgba(15,23,42,0.65)')
                .justifyContent(FlexAlign.Center)
                .backdropBlur(10)
              }
            }
            .width('100%')
            .padding(16)
          } else {
            // ── 初始空状态 ──
            Column({ space: 24 }) {
              Stack({ alignContent: Alignment.Center }) {
                Circle({ width: 120, height: 120 })
                  .fill('rgba(96,165,250,0.05)')
                Text('🔍')
                  .fontSize(56)
              }

              Column({ space: 8 }) {
                Text('等待图像输入')
                  .fontSize(22)
                  .fontWeight(FontWeight.Bold)
                  .fontColor('#F1F5F9')

                Text('点击下方按钮从相册选取或直接拍摄')
                  .fontSize(15)
                  .fontColor('#64748B')
              }
              .alignItems(HorizontalAlign.Center)
            }
            .width('100%')
            .constraintSize({ minHeight: '100%' })
            .justifyContent(FlexAlign.Center)
          }
        }
        .width('100%')
      }
      .width('100%')
      .layoutWeight(1)
      .scrollBar(BarState.Off)

      // ── 底部操作区 ──
      Row({ space: 16 }) {
        Button() {
          Text('🗑️').fontSize(22)
        }
        .width(56)
        .height(56)
        .type(ButtonType.Circle)
        .backgroundColor('rgba(255,255,255,0.1)')
        .onClick(() => {
          this.currentImageUri = '';
          this.detectionCount = 0;
          this.detectionLabels = [];
          this.canvasContext.clearRect(0, 0, 4000, 4000);
        })

        Button() {
          Row({ space: 8 }) {
            Text('📸').fontSize(20)
            Text('拍摄').fontSize(16).fontWeight(FontWeight.Bold)
          }
        }
        .layoutWeight(1)
        .height(56)
        .type(ButtonType.Normal)
        .borderRadius(16)
        .backgroundColor('#3B82F6')
        .shadow({ radius: 8, color: 'rgba(59,130,246,0.3)', offsetY: 4 })
        .onClick(() => {
          this.detectionCount = 0;
          this.detectionLabels = [];
          this.takePhoto();
        })

        Button() {
          Row({ space: 8 }) {
            Text('🖼️').fontSize(20)
            Text('相册').fontSize(16).fontWeight(FontWeight.Bold)
          }
        }
        .layoutWeight(1)
        .height(56)
        .type(ButtonType.Normal)
        .borderRadius(16)
        .backgroundColor('#8B5CF6')
        .shadow({ radius: 8, color: 'rgba(139,92,246,0.3)', offsetY: 4 })
        .onClick(() => {
          this.detectionCount = 0;
          this.detectionLabels = [];
          this.pickFromGallery();
        })
      }
      .width('100%')
      .padding({ left: 24, right: 24, top: 16, bottom: 32 })
      .backgroundColor('#0F172A')
      .shadow({ radius: 24, color: 'rgba(0,0,0,0.5)', offsetY: -10 })
    }
    .width('100%')
    .height('100%')
    .backgroundColor('#0F172A')
  }
}