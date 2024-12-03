package trajectorypredictiontransformer;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.Arrays;
import java.util.Random;

import static trajectorypredictiontransformer.TransformerArchitectureModel.*;

public class PointerwiseFeedforward {
    private int d_model;
    private int d_ff;
    private double dropout;
    private SDVariable w_1 = new SDVariable();
    private SDVariable w_2 = new SDVariable();
//    private INDArray w_1;
//    private INDArray w_2;
    private double scale;

    private int mRandomNumericalId;

    private Random mRandom;

    public static SameDiff sd = SameDiff.create();

    public PointerwiseFeedforward(SameDiff sd1, int d_model, int d_ff, double dropout) {
        this.sd = sd1;
        this.d_model = d_model;
        this.d_ff = d_ff;
        this.dropout = dropout;
//        this.w_1 = sd.var("w_1", Nd4j.zeros(batch_size, d_model, d_ff));
//        this.w_2 = sd.var("w_2", Nd4j.zeros(batch_size, d_ff, d_model));
//        this.w_1 = sd.var("w_1", Nd4j.zeros(d_model, d_ff));
//        this.w_2 = sd.var("w_2", Nd4j.zeros(d_ff, d_model));
        this.scale = Math.sqrt(2.0 / d_model);

    }

    public SDVariable forward(SDVariable x) {

        mRandom = new Random();
        mRandomNumericalId = mRandom.nextInt(10000);

        mRandom = new Random();
        mRandomNumericalId = mRandom.nextInt(10000);

        SDVariable w_1_2LinearOutput = linear(x, weights, bias);
        System.out.println(" PointerwiseFeedforward2 - forward - Arrays.toString(w_1_2LinearOutput.getShape()) - "+ Arrays.toString(w_1_2LinearOutput.getShape()));
        System.out.println(" PointerwiseFeedforward2 - forward - w_1_2LinearOutput.eval().shapeInfoToString() - "+ w_1_2LinearOutput.eval().shapeInfoToString());

        SDVariable w_1_2LinearOutputRelu = TransformerArchitectureModel.sd.nn.relu("w_1_2LinearOutputRelu"+" - "+mRandomNumericalId, w_1_2LinearOutput, 0.0);
        System.out.println(" PointerwiseFeedforward2 - forward - Arrays.toString(w_1_2LinearOutputRelu.getShape()) - "+ Arrays.toString(w_1_2LinearOutputRelu.getShape()));
        System.out.println(" PointerwiseFeedforward2 - forward - w_1_2LinearOutputRelu.eval().shapeInfoToString() - "+ w_1_2LinearOutputRelu.eval().shapeInfoToString());

        SDVariable w_1_2LinearOutputReluDropout = TransformerArchitectureModel.sd.nn.dropout("w_1_2LinearOutputReluDropout2"+" - "+mRandomNumericalId, w_1_2LinearOutputRelu, this.dropout);
        System.out.println(" PointerwiseFeedforward2 - forward - Arrays.toString(w_1_2LinearOutputReluDropout.getShape()) - "+ Arrays.toString(w_1_2LinearOutputReluDropout.getShape()));
        System.out.println(" PointerwiseFeedforward2 - forward - w_1_2LinearOutputReluDropout.eval().shapeInfoToString() - "+ w_1_2LinearOutputReluDropout.eval().shapeInfoToString());

        SDVariable w_2_2 = linear(w_1_2LinearOutputReluDropout, weights, bias);
        System.out.println(" PointerwiseFeedforward2 - forward - Arrays.toString(w_2_2.getShape()) - "+ Arrays.toString(w_2_2.getShape()));
        System.out.println(" PointerwiseFeedforward2 - forward - w_2_2.eval().shapeInfoToString() - "+ w_2_2.eval().shapeInfoToString());

//        SDVariable w_1x = x.mmul(w_1);
//        SDVariable reluOp = sd.nn.relu(w_1x, 0.0);
//        SDVariable dropoutOp = sd.nn.dropout("dropoutOp"+" - "+mRandomNumericalId, reluOp, this.dropout);
//        SDVariable w_2Relu = dropoutOp.mmul(w_2);
//        return w_2Relu;

        return w_2_2;
    }
}

