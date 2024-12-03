package trajectorypredictiontransformer;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;

import static trajectorypredictiontransformer.LocationNextNeuralNetworkV7_04.trainData;

public class LayerNorm3 {
    private SDVariable a2 = new SDVariable();
    private SDVariable b2 = new SDVariable();
//    private INDArray a2;
//    private INDArray b2;
    private double eps;

    private int features;

    public static SameDiff sd;

    Random mRandom;
    int mRandomNumericalId;

    public LayerNorm3(SameDiff sd, int features, double eps) {

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

        System.out.println(" - LayerNorm3 - Printing sd information");
        System.out.println(sd.summary());

        System.out.println(" - LayerNorm3 - Printing TransformerArchitectureModel.sd information");
        System.out.println(TransformerArchitectureModel.sd.summary());

    }

    public SDVariable forward(SDVariable x) {
//        public INDArray forward(INDArray x) {

//        sd.withNameScope("LayerNorm3");

//        SDVariable biasLocal = sd.constant("biasLayerNormTest", 0.5);
//        SDVariable gainLocal = sd.constant("biasLayerNormTest", 0.5);
//        SDVariable xNorm = sd.nn.layerNorm("xNormTransformerPredictor", linear0, gainLocal, biasLocal, false, 1);


        mRandom = new Random();
        mRandomNumericalId = mRandom.nextInt(1000000);

//        trainData.reset();
        HashMap<String, INDArray> placeholderData = new HashMap<>();
        placeholderData.put("input4", trainData.next().getFeatures());
        placeholderData.put("label4", trainData.next().getLabels());
        trainData.reset();

        System.out.println(" LayerNorm3 - Arrays.toString(x.getShape()) - "+ Arrays.toString(x.getShape()));
        System.out.println(" LayerNorm3 - x.eval(placeholderData).shapeInfoToString() - "+ x.eval(placeholderData).shapeInfoToString());

        System.out.println(" LayerNorm3 - Arrays.toString(x.getShape()) - "+ Arrays.toString(x.getShape()));
        System.out.println(" LayerNorm3 - x.eval(placeholderData).shapeInfoToString() - "+ x.eval(placeholderData).shapeInfoToString());
        System.out.println(" LayerNorm3 - x.eval(placeholderData) - "+ x.eval(placeholderData));

        INDArray xArray = x.eval(placeholderData);
        System.out.println(" LayerNorm3 - xArray.shape().length 1-  "+ xArray.shape().length);
        System.out.println(" LayerNorm3 - xArray.shape()[0] 1-  "+ xArray.shape()[0]);
        System.out.println(" LayerNorm3 - xArray.shape()[1] 1-  "+ xArray.shape()[1]);
        System.out.println(" LayerNorm3 - xArray.shape()[2] 1-  "+ xArray.shape()[2]);
        System.out.println(" LayerNorm3 - xArray 1-  "+ xArray);


        SDVariable mean = x.mean(true, -1);
//        SDVariable mean = x.mean(true, 1);
//        SDVariable mean = x.mean(false, 1);
//        SDVariable mean = x.mean(true, 2);

        SDVariable std = x.std(true, -1);
//        SDVariable std = x.std(true, 1, 2);
//        SDVariable std = x.std(true, 1);
//        SDVariable std = x.std(true);

//        SDVariable mean = x.mean(true, -1);
//        SDVariable std = x.std(true, -1);

        System.out.println(" LayerNorm3 - Arrays.toString(mean.getShape()) - "+ Arrays.toString(mean.getShape()));
        System.out.println(" LayerNorm3 - mean.eval(placeholderData).shapeInfoToString() - "+ mean.eval(placeholderData).shapeInfoToString());
        System.out.println(" LayerNorm3 - mean.eval(placeholderData) - "+ mean.eval(placeholderData));
        System.out.println(" LayerNorm3 - Arrays.toString(std.getShape()) - "+ Arrays.toString(std.getShape()));
        System.out.println(" LayerNorm3 - std.eval().shapeInfoToString() - "+ std.eval(placeholderData).shapeInfoToString());
        System.out.println(" LayerNorm3 - std.eval(placeholderData) - "+ std.eval(placeholderData));


        INDArray meanAfterBroadcastArray = mean.getArr().broadcast(x.eval(placeholderData).shape()[0], x.eval(placeholderData).shape()[1], x.eval(placeholderData).shape()[2]);
//        INDArray meanAfterBroadcastArray = mean.getArr().broadcast(x.getShape()[0], x.getShape()[1], x.getShape()[2]);
//        INDArray meanAfterBroadcastArray = mean.getArr().broadcast(x.eval(placeholderData).shape()[0], x.eval(placeholderData).shape()[1], x.eval(placeholderData).shape()[2]);
        System.out.println(" LayerNorm3 - meanAfterBroadcastArray.shape().length 1-  "+ meanAfterBroadcastArray.shape().length);
        System.out.println(" LayerNorm3 - meanAfterBroadcastArray.shape()[0] 1-  "+ meanAfterBroadcastArray.shape()[0]);
        System.out.println(" LayerNorm3 - meanAfterBroadcastArray.shape()[1] 1-  "+ meanAfterBroadcastArray.shape()[1]);
        System.out.println(" LayerNorm3 - meanAfterBroadcastArray.shape()[2] 1-  "+ meanAfterBroadcastArray.shape()[2]);

        SDVariable meanAfterBroadcast = sd.var(meanAfterBroadcastArray);
//        SDVariable meanAfterBroadcast = sd.var("meanAfterBroadcast"+" - "+mRandomNumericalId, meanAfterBroadcastArray);
        System.out.println(" LayerNorm3 - Arrays.toString(meanAfterBroadcast.getShape()) 1- "+ Arrays.toString(meanAfterBroadcast.getShape()));
        System.out.println(" LayerNorm3 - meanAfterBroadcast.eval(placeholderData).shapeInfoToString() 1- "+ meanAfterBroadcast.eval(placeholderData).shapeInfoToString());
        System.out.println(" LayerNorm3 - meanAfterBroadcast.eval(placeholderData) 1- "+ meanAfterBroadcast.eval(placeholderData));

//        mean.setVariableType(VariableType.CONSTANT);
//        std.setVariableType(VariableType.CONSTANT);

        INDArray xArrayResized = Nd4j.create(meanAfterBroadcast.getShape()[0], meanAfterBroadcast.getShape()[1], meanAfterBroadcast.getShape()[2]);
        INDArray xArrayResizedPopulated = xArrayResized.assign(xArray);
        System.out.println(" LayerNorm3 - xArrayResizedPopulated.shape().length 1-  "+ xArrayResizedPopulated.shape().length);
        System.out.println(" LayerNorm3 - xArrayResizedPopulated.shape()[0] 1-  "+ xArrayResizedPopulated.shape()[0]);
        System.out.println(" LayerNorm3 - xArrayResizedPopulated.shape()[1] 1-  "+ xArrayResizedPopulated.shape()[1]);
        System.out.println(" LayerNorm3 - xArrayResizedPopulated.shape()[2] 1-  "+ xArrayResizedPopulated.shape()[2]);
        System.out.println(" LayerNorm3 - xArrayResizedPopulated 1-  "+ xArrayResizedPopulated);


//        INDArray xArrayAfterBroadcast = xArray.broadcast(meanAfterBroadcast.getArr());
////        INDArray xArrayAfterBroadcast = xArray.broadcast(meanAfterBroadcast.getShape()[0], meanAfterBroadcast.getShape()[1], meanAfterBroadcast.getShape()[2]);
////        INDArray xArrayAfterBroadcast = xArray.broadcast(x.getShape()[0], x.getShape()[1], x.getShape()[2]);

        INDArray xSubMeanArray = xArrayResizedPopulated.sub(meanAfterBroadcastArray);
//        INDArray xSubMeanArray = xArray.sub(meanAfterBroadcastArray);
//        INDArray xSubMeanArray = xArrayAfterBroadcast.sub(meanAfterBroadcastArray);
        System.out.println(" LayerNorm3 - xSubMeanArray.shape().length 1-  "+ xSubMeanArray.shape().length);
        System.out.println(" LayerNorm3 - xSubMeanArray.shape()[0] 1-  "+ xSubMeanArray.shape()[0]);
        System.out.println(" LayerNorm3 - xSubMeanArray.shape()[1] 1-  "+ xSubMeanArray.shape()[1]);
        System.out.println(" LayerNorm3 - xSubMeanArray.shape()[2] 1-  "+ xSubMeanArray.shape()[2]);
        System.out.println(" LayerNorm3 - xSubMeanArray 1-  "+ xSubMeanArray);

        SDVariable xResizedPopulated = sd.var("xResizedPopulated"+" - "+mRandomNumericalId, xArrayResizedPopulated);
        System.out.println(" LayerNorm3 - Arrays.toString(xResizedPopulated.getShape()) 1- "+ Arrays.toString(xResizedPopulated.getShape()));
        System.out.println(" LayerNorm3 - xResizedPopulated.eval(placeholderData).shapeInfoToString() 1- "+ xResizedPopulated.eval(placeholderData).shapeInfoToString());
        System.out.println(" LayerNorm3 - xResizedPopulated.eval(placeholderData) 1- "+ xResizedPopulated.eval(placeholderData));
        System.out.println(" LayerNorm3 - Arrays.toString(meanAfterBroadcast.getShape()) 2- "+ Arrays.toString(meanAfterBroadcast.getShape()));
        System.out.println(" LayerNorm3 - meanAfterBroadcast.eval(placeholderData).shapeInfoToString() 2- "+ meanAfterBroadcast.eval(placeholderData).shapeInfoToString());
        System.out.println(" LayerNorm3 - meanAfterBroadcast.eval(placeholderData) 2- "+ meanAfterBroadcast.eval(placeholderData));

//        SDVariable meanAfterBroadcast = sd.var("meanAfterBroadcast", meanAfterBroadcastArray);

        SDVariable xSubMean = xResizedPopulated.sub(meanAfterBroadcast);
//        SDVariable xSubMean = sd.var(xResizedPopulated.sub(meanAfterBroadcast));
//        SDVariable xSubMean = sd.var(x.sub(meanAfterBroadcast));
//        SDVariable xSubMean = sd.var(x.sub(mean));
        System.out.println(" LayerNorm3 - Arrays.toString(xSubMean.getShape()) - "+ Arrays.toString(xSubMean.getShape()));
        System.out.println(" LayerNorm3 - xSubMean.eval().shapeInfoToString() - "+ xSubMean.eval(placeholderData).shapeInfoToString());
//        System.out.println(" LayerNorm3 - xSubMean.eval().shapeInfoToString() - "+ xSubMean.eval(placeholderData));

        SDVariable stdExpanded = sd.expandDims(std, 2);
        System.out.println(" LayerNorm3 - Arrays.toString(stdExpanded.getShape()) - "+ Arrays.toString(stdExpanded.getShape()));
        System.out.println(" LayerNorm3 - stdExpanded.eval().shapeInfoToString() - "+ stdExpanded.eval(placeholderData).shapeInfoToString());
//        System.out.println(" LayerNorm3 - stdExpanded.eval(placeholderData) - "+ stdExpanded.eval(placeholderData));

        INDArray stdExpandedAfterBroadcastArray = stdExpanded.eval(placeholderData).broadcast(xArray.shape()[0], xArray.shape()[1], xArray.shape()[2]);
//        INDArray stdExpandedAfterBroadcastArray = stdExpanded.getArr().broadcast(x.getShape()[0], x.getShape()[1], x.getShape()[2]);
//        INDArray stdExpandedAfterBroadcastArray = stdExpanded.getArr().broadcast(xSubMean.getShape()[0], xSubMean.getShape()[1], xSubMean.getShape()[2]);
        System.out.println(" LayerNorm3 - stdExpandedAfterBroadcastArray.shape().length 1-  "+ stdExpandedAfterBroadcastArray.shape().length);
        System.out.println(" LayerNorm3 - stdExpandedAfterBroadcastArray.shape()[0] 1-  "+ stdExpandedAfterBroadcastArray.shape()[0]);
        System.out.println(" LayerNorm3 - stdExpandedAfterBroadcastArray.shape()[1] 1-  "+ stdExpandedAfterBroadcastArray.shape()[1]);
        System.out.println(" LayerNorm3 - stdExpandedAfterBroadcastArray.shape()[2] 1-  "+ stdExpandedAfterBroadcastArray.shape()[2]);

        SDVariable stdExpandedAfterBroadcast = sd.var("stdExpandedAfterBroadcast"+" - "+mRandomNumericalId, meanAfterBroadcastArray);
        System.out.println(" LayerNorm3 - Arrays.toString(stdExpandedAfterBroadcast.getShape()) 1- "+ Arrays.toString(stdExpandedAfterBroadcast.getShape()));
        System.out.println(" LayerNorm3 - stdExpandedAfterBroadcast.eval(placeholderData).shapeInfoToString() 1- "+ stdExpandedAfterBroadcast.eval(placeholderData).shapeInfoToString());
//        System.out.println(" LayerNorm3 - stdExpandedAfterBroadcast.eval(placeholderData) 1- "+ stdExpandedAfterBroadcast.eval(placeholderData));

        a2 = sd.var(Nd4j.ones(1, 1, xSubMean.eval().shape()[2]));
        b2 = sd.var(Nd4j.zeros(xSubMean.eval().shape()[2]));

        System.out.println(" LayerNorm3 - Arrays.toString(a2.getShape()) - "+ Arrays.toString(a2.getShape()));
        System.out.println(" LayerNorm3 - a2.eval().shapeInfoToString() - "+ a2.eval(placeholderData).shapeInfoToString());
        System.out.println(" LayerNorm3 - Arrays.toString(xSubMean.getShape()) 1- "+ Arrays.toString(xSubMean.getShape()));
        System.out.println(" LayerNorm3 - xSubMean.eval().shapeInfoToString() 1- "+ xSubMean.eval(placeholderData).shapeInfoToString());
        System.out.println(" LayerNorm3 - Arrays.toString(b2.getShape()) - "+ Arrays.toString(b2.getShape()));
        System.out.println(" LayerNorm3 - b2.eval(placeholderData).shapeInfoToString() - "+ b2.eval(placeholderData).shapeInfoToString());

//        SDVariable norm = (xSubMean).div(stdExpandedAfterBroadcast.add(eps));
        SDVariable norm = (a2.mul(xSubMean)).div(stdExpandedAfterBroadcast.add(eps)).add(b2);
//        SDVariable norm = a2.mul(xSubMean).div(std.add(eps)).add(b2);
//        SDVariable norm = sd.var(x.sub(mean).div(std.add(eps)));
//        SDVariable norm = sd.var(a2.mul(x.sub(mean)).div(std.add(eps)).add(b2));
//        SDVariable norm = TransformerArchitectureModel.sd.var(a2.mul(x.sub(mean)).div(std.add(eps)).add(b2));

//        INDArray mean = x.mean(true, -1);
//        INDArray std = x.std(true, -1);
//        INDArray norm = a2.mul(x.sub(mean)).div(std.add(eps)).add(b2);

        System.out.println(" LayerNorm3 - Arrays.toString(norm.getShape()) - "+ Arrays.toString(norm.getShape()));
        System.out.println(" LayerNorm3 - norm.eval().shapeInfoToString() - "+ norm.eval(placeholderData).shapeInfoToString());
        System.out.println(" LayerNorm3 - norm.eval(placeholderData) - "+ norm.eval(placeholderData));

        SDVariable meanLayerNorm3 = norm.mean(-1);
        System.out.println(" LayerNorm3 - Arrays.toString(meanLayerNorm3.getShape()) - "+ Arrays.toString(meanLayerNorm3.getShape()));
        System.out.println(" LayerNorm3 - meanLayerNorm3.eval().shapeInfoToString() - "+ meanLayerNorm3.eval(placeholderData).shapeInfoToString());
        System.out.println(" LayerNorm3 - meanLayerNorm3.eval(placeholderData) - "+ meanLayerNorm3.eval(placeholderData));

        return norm;
//        return x;   //IN ORDER TO USE THIS NEED TO CHANGE INPUT IN MultiHeadAttention2
    }
}

