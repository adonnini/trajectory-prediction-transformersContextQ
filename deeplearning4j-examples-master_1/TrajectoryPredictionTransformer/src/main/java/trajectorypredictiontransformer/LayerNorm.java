package trajectorypredictiontransformer;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;

import static trajectorypredictiontransformer.LocationNextNeuralNetworkV7_04.trainData;

public class LayerNorm {
    private SDVariable a2 = new SDVariable();
    private SDVariable b2 = new SDVariable();
//    private INDArray a2;
//    private INDArray b2;
    private double eps;

    private int features;

    public static SameDiff sd;

    Random mRandom;
    int mRandomNumericalId;

    public LayerNorm(SameDiff sd, int features, double eps) {

//        this.sd = TransformerArchitectureModel.sd;
        this.sd = sd;
        this.eps = eps;
        this.features = features;

//        a2 = sd.var(Nd4j.ones(features));
//        b2 = sd.var(Nd4j.zeros(features));

//        LayerNorm.sd.associateArrayWithVariable(Nd4j.ones(features), a2);
//        LayerNorm.sd.associateArrayWithVariable(Nd4j.zeros(features), b2);
//        TransformerArchitectureModel.sd.associateArrayWithVariable(Nd4j.ones(features), a2);
//        TransformerArchitectureModel.sd.associateArrayWithVariable(Nd4j.zeros(features), b2);
//        this.a2 = a2.setArray(Nd4j.ones(features));
//        this.b2 = b2.setArray(Nd4j.zeros(features));

//        this.a2 = Nd4j.ones(features);
//        this.b2 = Nd4j.zeros(features);

        System.out.println(" - LayerNorm - Printing sd information");
        System.out.println(sd.summary());

        System.out.println(" - LayerNorm - Printing TransformerArchitectureModel.sd information");
        System.out.println(TransformerArchitectureModel.sd.summary());

    }

    public SDVariable forward(SDVariable x) {
//        public INDArray forward(INDArray x) {

        System.out.println(" LayerNorm - forward - About to start processing --- ");

//        SDVariable biasLocal = sd.constant("biasLayerNormTest", 0.5);
//        SDVariable gainLocal = sd.constant("biasLayerNormTest", 0.5);
//        SDVariable xNorm = sd.nn.layerNorm("xNormTransformerPredictor", linear0, gainLocal, biasLocal, false, 1);

//        sd.withNameScope("LayerNorm");

        mRandom = new Random();
        mRandomNumericalId = mRandom.nextInt(10000);

//        trainData.reset();
        HashMap<String, INDArray> placeholderData = new HashMap<>();
        placeholderData.put("input4", trainData.next().getFeatures());
        placeholderData.put("label4", trainData.next().getLabels());
        trainData.reset();

        INDArray xArray = x.eval(placeholderData);
        SDVariable mean = x.mean(true, -1);
        SDVariable std = x.std(true, -1);

        INDArray meanArray = mean.eval(placeholderData);

        INDArray stdArray = std.eval();
//        INDArray stdArray = std.eval(placeholderData);


//        System.out.println(" LayerNorm - Arrays.toString(x.getShape()) - "+ Arrays.toString(x.getShape()));
//        System.out.println(" LayerNorm - x.eval(placeholderData).shapeInfoToString() - "+ x.eval(placeholderData).shapeInfoToString());

        System.out.println(" LayerNorm - Arrays.toString(x.getShape()) - "+ Arrays.toString(x.getShape()));
        System.out.println(" LayerNorm - xArray.shapeInfoToString() - "+ xArray.shapeInfoToString());
//        System.out.println(" LayerNorm - x.eval(placeholderData).shapeInfoToString() - "+ x.eval(placeholderData).shapeInfoToString());
        System.out.println(" LayerNorm - xArray - "+ xArray);
//        System.out.println(" LayerNorm - x.eval(placeholderData) - "+ x.eval(placeholderData));

//        INDArray xArray = x.eval(placeholderData);
        System.out.println(" LayerNorm - xArray.shape().length 1-  "+ xArray.shape().length);
        System.out.println(" LayerNorm - xArray.shape()[0] 1-  "+ xArray.shape()[0]);
        System.out.println(" LayerNorm - xArray.shape()[1] 1-  "+ xArray.shape()[1]);
        System.out.println(" LayerNorm - xArray.shape()[2] 1-  "+ xArray.shape()[2]);
        System.out.println(" LayerNorm - xArray 1-  "+ xArray);

//        SDVariable mean = x.mean(true, -1);
//        SDVariable mean = x.mean(true, 1);
//        SDVariable mean = x.mean(false, 1);
//        SDVariable mean = x.mean(true, 2);

//        INDArray meanArray = mean.eval(placeholderData);

//        SDVariable std = x.std(true, -1);
//        SDVariable std = x.std(true, 1, 2);
//        SDVariable std = x.std(true, 1);
//        SDVariable std = x.std(true);

//        INDArray stdArray = std.eval(placeholderData);

//        SDVariable mean = x.mean(true, -1);
//        SDVariable std = x.std(true, -1);


        System.out.println(" LayerNorm - Arrays.toString(mean.getShape()) - "+ Arrays.toString(mean.getShape()));
        System.out.println(" LayerNorm - meanArray.shapeInfoToString() - "+ meanArray.shapeInfoToString());
        System.out.println(" LayerNorm - meanArray - "+ meanArray);
        System.out.println(" LayerNorm - Arrays.toString(std.getShape()) - "+ Arrays.toString(std.getShape()));
        System.out.println(" LayerNorm - stdArray.shapeInfoToString() - "+ stdArray.shapeInfoToString());
        System.out.println(" LayerNorm - stdArray - "+ stdArray);

        ////

//        xArray = x.eval(placeholderData);
        System.out.println(" LayerNorm - xArray - 2 - "+ xArray);

        INDArray meanAfterBroadcastArray = mean.getArr().broadcast(xArray.shape()[0], xArray.shape()[1], xArray.shape()[2]);
//        INDArray meanAfterBroadcastArray = mean.getArr().broadcast(x.eval(placeholderData).shape()[0], x.eval(placeholderData).shape()[1], x.eval(placeholderData).shape()[2]);
//        INDArray meanAfterBroadcastArray = mean.getArr().broadcast(x.getShape()[0], x.getShape()[1], x.getShape()[2]);
//        INDArray meanAfterBroadcastArray = mean.getArr().broadcast(x.eval(placeholderData).shape()[0], x.eval(placeholderData).shape()[1], x.eval(placeholderData).shape()[2]);
        System.out.println(" LayerNorm - meanAfterBroadcastArray.shape().length 1-  "+ meanAfterBroadcastArray.shape().length);
        System.out.println(" LayerNorm - meanAfterBroadcastArray.shape()[0] 1-  "+ meanAfterBroadcastArray.shape()[0]);
        System.out.println(" LayerNorm - meanAfterBroadcastArray.shape()[1] 1-  "+ meanAfterBroadcastArray.shape()[1]);
        System.out.println(" LayerNorm - meanAfterBroadcastArray.shape()[2] 1-  "+ meanAfterBroadcastArray.shape()[2]);

        mRandomNumericalId = mRandom.nextInt(100000);
        SDVariable meanAfterBroadcast = sd.var("meanAfterBroadcast"+mRandomNumericalId,meanAfterBroadcastArray);
//        INDArray meanAfterBroadcastArray2 = meanAfterBroadcast.eval(placeholderData);
//        SDVariable meanAfterBroadcast = sd.var("meanAfterBroadcast"+" - "+mRandomNumericalId, meanAfterBroadcastArray);
//        try {
//            System.out.println(" LayerNorm - Arrays.toString(meanAfterBroadcast.getShape()) 1- "+ Arrays.toString(meanAfterBroadcast.getShape()));
//            System.out.println(" LayerNorm -meanAfterBroadcastArray2.shapeInfoToString() 1- "+ meanAfterBroadcastArray2.shapeInfoToString());
//            System.out.println(" LayerNorm - meanAfterBroadcastArray2 1- "+ meanAfterBroadcastArray2);
//            System.out.println(" LayerNorm - meanAfterBroadcast.eval(placeholderData).shapeInfoToString() 1- "+ meanAfterBroadcast.eval(placeholderData).shapeInfoToString());
//            System.out.println(" LayerNorm - meanAfterBroadcast.eval(placeholderData) 1- "+ meanAfterBroadcast.eval(placeholderData));
//        } catch (Exception e) {
//            throw new RuntimeException(e);
//        }

//        mean.setVariableType(VariableType.CONSTANT);
//        std.setVariableType(VariableType.CONSTANT);

        INDArray xArrayResized = Nd4j.create(meanAfterBroadcast.getShape()[0], meanAfterBroadcast.getShape()[1], meanAfterBroadcast.getShape()[2]);
        INDArray xArrayResizedPopulated = xArrayResized.assign(xArray);
        System.out.println(" LayerNorm - xArrayResizedPopulated.shape().length 1-  "+ xArrayResizedPopulated.shape().length);
        System.out.println(" LayerNorm - xArrayResizedPopulated.shape()[0] 1-  "+ xArrayResizedPopulated.shape()[0]);
        System.out.println(" LayerNorm - xArrayResizedPopulated.shape()[1] 1-  "+ xArrayResizedPopulated.shape()[1]);
        System.out.println(" LayerNorm - xArrayResizedPopulated.shape()[2] 1-  "+ xArrayResizedPopulated.shape()[2]);


//        INDArray xArrayAfterBroadcast = xArray.broadcast(meanAfterBroadcast.getArr());
////        INDArray xArrayAfterBroadcast = xArray.broadcast(meanAfterBroadcast.getShape()[0], meanAfterBroadcast.getShape()[1], meanAfterBroadcast.getShape()[2]);
////        INDArray xArrayAfterBroadcast = xArray.broadcast(x.getShape()[0], x.getShape()[1], x.getShape()[2]);

        INDArray xSubMeanArray = xArrayResizedPopulated.sub(meanAfterBroadcastArray);
//        INDArray xSubMeanArray = xArray.sub(meanAfterBroadcastArray);
//        INDArray xSubMeanArray = xArrayAfterBroadcast.sub(meanAfterBroadcastArray);
        System.out.println(" LayerNorm - xSubMeanArray.shape().length 1-  "+ xSubMeanArray.shape().length);
        System.out.println(" LayerNorm - xSubMeanArray.shape()[0] 1-  "+ xSubMeanArray.shape()[0]);
        System.out.println(" LayerNorm - xSubMeanArray.shape()[1] 1-  "+ xSubMeanArray.shape()[1]);
        System.out.println(" LayerNorm - xSubMeanArray.shape()[2] 1-  "+ xSubMeanArray.shape()[2]);
        System.out.println(" LayerNorm - xSubMeanArray 1-  "+ xSubMeanArray);

        SDVariable xResizedPopulated = sd.var("xResizedPopulated"+" - "+mRandomNumericalId, xArrayResizedPopulated);
//        System.out.println(" LayerNorm - Arrays.toString(xResizedPopulated.getShape()) 1- "+ Arrays.toString(xResizedPopulated.getShape()));
//        System.out.println(" LayerNorm - xResizedPopulated.eval(placeholderData).shapeInfoToString() 1- "+ xResizedPopulated.eval(placeholderData).shapeInfoToString());
//        System.out.println(" LayerNorm - xResizedPopulated.eval(placeholderData) 1- "+ xResizedPopulated.eval(placeholderData));
//        System.out.println(" LayerNorm - Arrays.toString(meanAfterBroadcast.getShape()) 2- "+ Arrays.toString(meanAfterBroadcast.getShape()));
//        System.out.println(" LayerNorm - meanAfterBroadcast.eval(placeholderData).shapeInfoToString() 2- "+ meanAfterBroadcast.eval(placeholderData).shapeInfoToString());
//        System.out.println(" LayerNorm - meanAfterBroadcast.eval(placeholderData) 2- "+ meanAfterBroadcast.eval(placeholderData));

//        SDVariable meanAfterBroadcast = sd.var("meanAfterBroadcast", meanAfterBroadcastArray);

        SDVariable xSubMean = xResizedPopulated.sub(meanAfterBroadcast);
//        SDVariable xSubMean = sd.var(xResizedPopulated.sub(meanAfterBroadcast));
//        SDVariable xSubMean = sd.var(x.sub(meanAfterBroadcast));
//        SDVariable xSubMean = sd.var(x.sub(mean));
//        System.out.println(" LayerNorm - Arrays.toString(xSubMean.getShape()) - "+ Arrays.toString(xSubMean.getShape()));
//        System.out.println(" LayerNorm - xSubMean.eval().shapeInfoToString() - "+ xSubMean.eval(placeholderData).shapeInfoToString());
//        System.out.println(" LayerNorm - xSubMean.eval(placeholderData) - "+ xSubMean.eval(placeholderData));

//        SDVariable stdExpanded = sd.expandDims(std, 2);
//        System.out.println(" LayerNorm - Arrays.toString(stdExpanded.getShape()) - "+ Arrays.toString(stdExpanded.getShape()));
//        System.out.println(" LayerNorm - stdExpanded.eval().shapeInfoToString() - "+ stdExpanded.eval(placeholderData).shapeInfoToString());
//        System.out.println(" LayerNorm - stdExpanded.eval().shapeInfoToString() - "+ stdExpanded.eval(placeholderData));

//        INDArray stdExpandedAfterBroadcastArray = stdExpanded.getArr().broadcast(x.eval().shape()[0], x.eval().shape()[1], x.eval().shape()[2]);
//        INDArray stdExpandedAfterBroadcastArray = stdExpanded.getArr().broadcast(x.getShape()[0], x.getShape()[1], x.getShape()[2]);
//        INDArray stdExpandedAfterBroadcastArray = stdExpanded.getArr().broadcast(xSubMean.getShape()[0], xSubMean.getShape()[1], xSubMean.getShape()[2]);
//        System.out.println(" LayerNorm - stdExpandedAfterBroadcastArray.shape().length 1-  "+ stdExpandedAfterBroadcastArray.shape().length);
//        System.out.println(" LayerNorm - stdExpandedAfterBroadcastArray.shape()[0] 1-  "+ stdExpandedAfterBroadcastArray.shape()[0]);
//        System.out.println(" LayerNorm - stdExpandedAfterBroadcastArray.shape()[1] 1-  "+ stdExpandedAfterBroadcastArray.shape()[1]);
//        System.out.println(" LayerNorm - stdExpandedAfterBroadcastArray.shape()[2] 1-  "+ stdExpandedAfterBroadcastArray.shape()[2]);

        SDVariable stdExpandedAfterBroadcast = sd.var("stdExpandedAfterBroadcast"+" - "+mRandomNumericalId, meanAfterBroadcastArray);
//        System.out.println(" LayerNorm - Arrays.toString(stdExpandedAfterBroadcast.getShape()) 1- "+ Arrays.toString(stdExpandedAfterBroadcast.getShape()));
//        System.out.println(" LayerNorm - stdExpandedAfterBroadcast.eval(placeholderData).shapeInfoToString() 1- "+ stdExpandedAfterBroadcast.eval(placeholderData).shapeInfoToString());
//        System.out.println(" LayerNorm - stdExpandedAfterBroadcast.eval(placeholderData) 1- "+ stdExpandedAfterBroadcast.eval(placeholderData));

        mRandomNumericalId = mRandom.nextInt(100000);
        a2 = sd.var("a2"+mRandomNumericalId, Nd4j.ones(1, 1, xSubMean.eval().shape()[2]));
        b2 = sd.var("b2"+mRandomNumericalId, Nd4j.zeros(xSubMean.eval().shape()[2]));

//        System.out.println(" LayerNorm - Arrays.toString(a2.getShape()) - "+ Arrays.toString(a2.getShape()));
//        System.out.println(" LayerNorm - a2.eval().shapeInfoToString() - "+ a2.eval(placeholderData).shapeInfoToString());
//        System.out.println(" LayerNorm - Arrays.toString(xSubMean.getShape()) 1- "+ Arrays.toString(xSubMean.getShape()));
//        System.out.println(" LayerNorm - xSubMean.eval().shapeInfoToString() 1- "+ xSubMean.eval(placeholderData).shapeInfoToString());
//        System.out.println(" LayerNorm - Arrays.toString(b2.getShape()) - "+ Arrays.toString(b2.getShape()));
//        System.out.println(" LayerNorm - b2.eval().shapeInfoToString() - "+ b2.eval(placeholderData).shapeInfoToString());

//        SDVariable norm = (xSubMean).div(stdExpandedAfterBroadcast.add(eps));
        SDVariable norm = (a2.mul(xSubMean)).div(stdExpandedAfterBroadcast.add(eps)).add(b2);
//        SDVariable norm = a2.mul(xSubMean).div(std.add(eps)).add(b2);
//        SDVariable norm = sd.var(x.sub(mean).div(std.add(eps)));
//        SDVariable norm = sd.var(a2.mul(x.sub(mean)).div(std.add(eps)).add(b2));
//        SDVariable norm = TransformerArchitectureModel.sd.var(a2.mul(x.sub(mean)).div(std.add(eps)).add(b2));

//        INDArray mean = x.mean(true, -1);
//        INDArray std = x.std(true, -1);
//        INDArray norm = a2.mul(x.sub(mean)).div(std.add(eps)).add(b2);

//        System.out.println(" LayerNorm - Arrays.toString(norm.getShape()) - "+ Arrays.toString(norm.getShape()));
//        System.out.println(" LayerNorm - norm.eval().shapeInfoToString() - "+ norm.eval(placeholderData).shapeInfoToString());
//        System.out.println(" LayerNorm - norm.eval(placeholderData) - "+ norm.eval(placeholderData));

//        SDVariable meanLayerNorm = norm.mean(-1);
//        System.out.println(" LayerNorm - Arrays.toString(meanLayerNorm.getShape()) - "+ Arrays.toString(meanLayerNorm.getShape()));
//        System.out.println(" LayerNorm - meanLayerNorm.eval().shapeInfoToString() - "+ meanLayerNorm.eval(placeholderData).shapeInfoToString());
//        System.out.println(" LayerNorm - meanLayerNorm.eval(placeholderData) - "+ meanLayerNorm.eval(placeholderData));

        ////

        System.out.println(" LayerNorm - forward - Completed processing --- ");

        return norm;
//        return x;
    }
}

