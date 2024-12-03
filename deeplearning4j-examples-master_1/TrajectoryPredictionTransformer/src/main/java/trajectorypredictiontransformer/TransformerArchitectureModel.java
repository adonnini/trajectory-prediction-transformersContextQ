package trajectorypredictiontransformer;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.weightinit.impl.XavierInitScheme;

import java.util.*;

import static trajectorypredictiontransformer.LocationNextNeuralNetworkV7_04.*;

public class TransformerArchitectureModel {

    public static int encoderIpSize;
    public static int decoderIpSize;
    public static int modelOpSize;
    public static int embSize;
    public static int numHeads;
    public static int ffHiddenSize;
    public static int n;
    public static double dropout;
    public static int batch_size;
    public static int labelCount;

    public static SameDiff sd = SameDiff.create();

    public static SDVariable weights = new SDVariable();

    public static SDVariable bias = new SDVariable();

    public static Random mRandom;
    public static int mRandomNumericalId;

    public static SDVariable linear(SDVariable input, SDVariable weights, SDVariable bias) {

        mRandom = new Random();
        mRandomNumericalId = mRandom.nextInt(100000);

        input.setDataType(DataType.FLOAT);
        weights.setDataType(DataType.FLOAT);
        bias.setDataType(DataType.FLOAT);

        System.out.println(" linear - input.eval().shapeInfoToString() - "+ input.eval().shapeInfoToString());
        System.out.println(" linear - weights.eval().shapeInfoToString() - "+ weights.eval().shapeInfoToString());
        System.out.println(" linear - bias.eval().shapeInfoToString() - "+ bias.eval().shapeInfoToString());
        System.out.println(" linear - input.eval() - "+ input.eval());
        System.out.println(" linear - weights.eval() - "+ weights.eval());
        System.out.println(" linear - bias.eval() - "+ bias.eval());

        SDVariable inputPermuted = sd.permute(input, 0, 2, 1);
        System.out.println(" linear - ======================================================= - ");
        System.out.println(" linear - Arrays.toString(inputPermuted.eval().shape()) - "+ Arrays.toString(inputPermuted.eval().shape()));
        System.out.println(" linear - inputPermuted.eval() - "+ inputPermuted.eval());

        SDVariable weightsResizedPopulated = new SDVariable();
        SDVariable mmulOutput = new SDVariable();

        if(inputPermuted.eval().shape()[2] == embSize) {
            INDArray weightsArray = weights.getArr();
            INDArray weightsArrayResized = Nd4j.create(inputPermuted.eval().shape()[0], inputPermuted.eval().shape()[1], inputPermuted.eval().shape()[2]);
            INDArray weightsArrayResizedPopulated = weightsArrayResized.assign(weightsArray);
            weightsResizedPopulated = sd.var("weightsResizedPopulated"+" - "+mRandomNumericalId, weightsArrayResizedPopulated);
            System.out.println(" linear - weightsArrayResizedPopulated - "+ weightsArrayResizedPopulated);

            inputPermuted.setDataType(DataType.FLOAT);
            weights.setDataType(DataType.FLOAT);
            bias.setDataType(DataType.FLOAT);
            mmulOutput = inputPermuted.mmul(weightsResizedPopulated).add(bias);
            System.out.println(" linear - Arrays.toString(inputPermuted.getShape()) - "+ Arrays.toString(inputPermuted.getShape()));
            System.out.println(" linear - inputPermuted.eval().shapeInfoToString() - "+ inputPermuted.eval().shapeInfoToString());
            System.out.println(" linear - Arrays.toString(weightsResizedPopulated.getShape()) - "+ Arrays.toString(weightsResizedPopulated.getShape()));
            System.out.println(" linear - weightsResizedPopulated.eval().shapeInfoToString() - "+ weightsResizedPopulated.eval().shapeInfoToString());
            System.out.println(" linear - Arrays.toString(mmulOutput.getShape()) - "+ Arrays.toString(mmulOutput.getShape()));
            System.out.println(" linear - mmulOutput.eval().shapeInfoToString() - "+ mmulOutput.eval().shapeInfoToString());
            System.out.println(" linear - mmulOutput.eval() 0- "+ mmulOutput.eval());

        }
        else {
            inputPermuted.setDataType(DataType.FLOAT);
            weights.setDataType(DataType.FLOAT);
            bias.setDataType(DataType.FLOAT);
            mmulOutput = inputPermuted.mmul(weights).add(bias);
            System.out.println(" linear - Arrays.toString(inputPermuted.getShape()) - "+ Arrays.toString(inputPermuted.getShape()));
            System.out.println(" linear - inputPermuted.eval().shapeInfoToString() - "+ inputPermuted.eval().shapeInfoToString());
            System.out.println(" linear - Arrays.toString(weights.getShape()) - "+ Arrays.toString(weights.getShape()));
            System.out.println(" linear - weights.eval().shapeInfoToString() - "+ weights.eval().shapeInfoToString());
            System.out.println(" linear - Arrays.toString(mmulOutput.getShape()) - "+ Arrays.toString(mmulOutput.getShape()));
            System.out.println(" linear - mmulOutput.eval().shapeInfoToString() - "+ mmulOutput.eval().shapeInfoToString());
            System.out.println(" linear - mmulOutput.eval() 1- "+ mmulOutput.eval());

        }

//        inputPermuted.setDataType(DataType.FLOAT);
//        weights.setDataType(DataType.FLOAT);
//        bias.setDataType(DataType.FLOAT);
//        mmulOutput = inputPermuted.mmul(weights).add(bias);
//        System.out.println(" linear - Arrays.toString(mmulOutput.getShape()) - "+ Arrays.toString(mmulOutput.getShape()));
//        System.out.println(" linear - mmulOutput.eval().shapeInfoToString() - "+ mmulOutput.eval().shapeInfoToString());

//        SDVariable mmulOutputPermuted = sd.permute("linearOutput", mmulOutput, 0, 2, 1);
//        System.out.println(" linear - Arrays.toString(mmulOutputPermuted.getShape()) - "+ Arrays.toString(mmulOutputPermuted.getShape()));
//        System.out.println(" linear - mmulOutputPermuted.eval().shapeInfoToString() - "+ mmulOutputPermuted.eval().shapeInfoToString());

//        input.setDataType(DataType.FLOAT);
//        System.out.println(" linear - Arrays.toString(sd.nn.linear(input, weights, bias).getShape()) - "+ Arrays.toString(sd.nn.linear(input, weights, bias).getShape()));
//        System.out.println(" linear - sd.nn.linear(input, weights, bias).eval().shapeInfoToString() - "+ sd.nn.linear(input, weights, bias).eval().shapeInfoToString());

        return mmulOutput;
//        return sd.nn.linear(input, weights, bias);
    }

    public static SDVariable dropout(SDVariable input, double dropout) {

        return sd.nn.dropout(input, dropout);
    }

    public static SDVariable sequential(SDVariable input, double dropout)
    {
        Random mRandom;
        int mRandomNumericalId;

        mRandom = new Random();
        mRandomNumericalId = mRandom.nextInt(1000000);

        SDVariable linear1 = linear(input, weights, bias);
        SDVariable relu = sd.nn.relu("relu"+" - "+mRandomNumericalId, linear1, 0);
        SDVariable dropoutLayer = sd.nn().dropout("dropout"+" - "+mRandomNumericalId, relu, dropout);
        SDVariable linear2 = linear(dropoutLayer, weights, bias);

        return linear2;

    }

//    public static class LayerNorm {
//        private SDVariable a2 = new SDVariable();
//        private SDVariable b2 = new SDVariable();
//        //    private INDArray a2;
////    private INDArray b2;
//        private double eps;
//
////        public static SameDiff sd;
//
////        public LayerNorm(int features, double eps) {
//////        public LayerNorm(SameDiff sd, int features, double eps) {
////
//////            this.sd = TransformerArchitectureModel.sd;
//////        this.sd = sd;
////            this.eps = eps;
////
////            sd.associateArrayWithVariable(Nd4j.ones(features), a2);
////            sd.associateArrayWithVariable(Nd4j.zeros(features), b2);
//////        LayerNorm.sd.associateArrayWithVariable(Nd4j.ones(features), a2);
//////        LayerNorm.sd.associateArrayWithVariable(Nd4j.zeros(features), b2);
//////            TransformerArchitectureModel.sd.associateArrayWithVariable(Nd4j.ones(features), a2);
//////            TransformerArchitectureModel.sd.associateArrayWithVariable(Nd4j.zeros(features), b2);
//////        this.a2 = a2.setArray(Nd4j.ones(features));
//////        this.b2 = b2.setArray(Nd4j.zeros(features));
////
//////        this.a2 = Nd4j.ones(features);
//////        this.b2 = Nd4j.zeros(features);
////
////            System.out.println(" - LayerNorm - Printing sd information");
////            System.out.println(sd.summary());
////
////            System.out.println(" - LayerNorm - Printing TransformerArchitectureModel.sd information");
////            System.out.println(TransformerArchitectureModel.sd.summary());
////
////        }
//
//        public SDVariable forward(SDVariable x) {
////        public INDArray forward(INDArray x) {
//            SDVariable mean = x.mean(true, -1);
//            SDVariable std = x.std(true, -1);
//            SDVariable norm = sd.var(a2.mul(x.sub(mean)).div(std.add(eps)).add(b2));
////        SDVariable norm = TransformerArchitectureModel.sd.var(a2.mul(x.sub(mean)).div(std.add(eps)).add(b2));
//
////        INDArray mean = x.mean(true, -1);
////        INDArray std = x.std(true, -1);
////        INDArray norm = a2.mul(x.sub(mean)).div(std.add(eps)).add(b2);
//
//            return norm;
//        }
//    }

    // Multi Head Attention Layer
    public static class MultiHeadAttention {

//    Class to create the multi head attention layer for
//    encoder and decoder
//
//        Class constructor
//
//        INPUT:
//        num_head - (int) number of heads in multi head attention layer
//        emb_size - (int) embedding size of the data
//        dropout - (float) dropout percentage. Default value = 0.1

//        private int embSize;
//        private int numHeads;
//        private double dropout;

//        private SameDiff sd;

        SDVariable qLinearWeights = new SDVariable();
        SDVariable kLinearWeights = new SDVariable();
        SDVariable vLinearWeights = new SDVariable();
        SDVariable postAttWeights = new SDVariable();
        SDVariable postAttBias = new SDVariable();
        SDVariable dropoutLayer = new SDVariable();


        public MultiHeadAttention(int numHeads1, int embSize1, double dropout1) {

            embSize = embSize1;
            numHeads = numHeads1;
            dropout = dropout1;
//            this.embSize = embSize;
//            this.numHeads = numHeads;
//            this.dropout = dropout;

            System.out.println("embSize: " + embSize);
            System.out.println("numHeads: " + numHeads);

            // making sure that the embedding size is divisible by the number of heads
            if (embSize % numHeads != 0) {
                throw new IllegalArgumentException("Embedding size must be divisible by the number of heads.");
            }

//            INDArray arr = Nd4j.create(embSize,embSize);
//            SDVariable input = sd.var("input", arr);
//
//            SDVariable q_linear = linear(input, weights, bias);
//            SDVariable k_linear = linear(input, weights, bias);
//            SDVariable v_linear = linear(input, weights, bias);
//
//            SDVariable post_att = linear(input, weights, bias);
//
//            SDVariable dropoutSDV = dropout(input, dropout);

        }

        public SDVariable forward(SameDiff sd, SDVariable Q, SDVariable K, SDVariable V, SDVariable mask) {
//            public INDArray forward(INDArray Q, INDArray K, INDArray V, INDArray mask) {

//           forward function for MultiHeadAttention
//
//           INPUT:
//           Q - (torch tensor) query for the transformer. Shape = (B, N, C)
//           K - (torch tensor) keys for the transformer. Shape = (B, N, C)
//           V - (torch tensor) values for the transformer. Shape = (B, N, C)
//           mask - (torch tensor) mask for decoder multi head attention layer
//
//           OUTPUT:
//           att_output - (torch tensor) output of the multi head attention layer. Shape = (B, N, C)

            Random mRandom = new Random();
            int mRandomNumericalId = mRandom.nextInt(100000);

            System.out.println(" MultiHeadAttention - forward - Arrays.toString(Q.getShape()) - "+ Arrays.toString(Q.getShape()));
            System.out.println(" MultiHeadAttention - forward - Q.eval().shapeInfoToString() - "+ Q.eval().shapeInfoToString());

            // creating MLP layer for post-attention
            postAttWeights = sd.var(Nd4j.randn(batch_size, embSize, embSize));
            postAttBias = sd.var(Nd4j.zeros(1, embSize));

            // creating dropout layer
            dropoutLayer = sd.var(Nd4j.scalar(dropout));

            // Same mask applied to all h heads.
            if (mask != null) {
                sd.expandDims(mask, 1);
//                mask = mask.reshape(mask.size(0), 1, mask.size(1), mask.size(2));
            }

//            qLinearWeights = sd.expandDims(qLinearWeights, 2);
//            kLinearWeights = sd.expandDims(kLinearWeights, 2);
//            vLinearWeights = sd.expandDims(vLinearWeights, 2);

//        # passing the Q, K, and V through 1 layer MLP
//        Q, K, V = self.q_linear(Q), self.k_linear(K), self.v_linear(V)  # Shape = (B, N, C)

            SDVariable weights4 = sd.var("weights4"+" - "+mRandomNumericalId, new XavierInitScheme('c', encoder_ip_size, model_op_size), DataType.FLOAT, Q.eval().shape()[0], Q.eval().shape()[1], Q.eval().shape()[2]);

            SDVariable qLinear = linear(Q, weights4, bias);
            SDVariable kLinear = linear(K, weights4, bias);
            SDVariable vLinear = linear(V, weights4, bias);

//        # splitting Q, K and V based on num_heads
//        batch_size = Q.shape[0]
//        new_emb_size = self.emb_size // self.num_heads

            // passing the Q, K, and V through 1 layer MLP

            // splitting Q, K, and V based on num_heads
            int batchSize = (int) Q.eval().shape()[0];
//            int batchSize = (int) Q.getArr().shape()[0];
//            int batchSize = Q.size(0);
            int newEmbSize = embSize / numHeads;
            System.out.println(" MultiHeadAttention - forward - batchSize - "+ batchSize);
            System.out.println(" MultiHeadAttention - forward - newEmbSize - "+ newEmbSize);

//        Q = Q.reshape(batch_size, -1, self.num_heads, new_emb_size)     # Shape = (B, N, H, C//H)
//        K = K.reshape(batch_size, -1, self.num_heads, new_emb_size)     # Shape = (B, N, H, C//H)
//        V = V.reshape(batch_size, -1, self.num_heads, new_emb_size)     # Shape = (B, N, H, C//H)

            INDArray QArray = qLinear.getArr();
            INDArray QArrayNewDimensions = Nd4j.create(batchSize, Q.eval().shape()[2], numHeads, newEmbSize);
            INDArray QArrayNewDimensionsPopulated = QArrayNewDimensions.assign(QArray);
            SDVariable QNewDimensions = sd.var(QArrayNewDimensionsPopulated);

            INDArray KArray = kLinear.getArr();
            INDArray KArrayNewDimensions = Nd4j.create(batchSize, K.eval().shape()[2], numHeads, newEmbSize);
            INDArray KArrayNewDimensionsPopulated = KArrayNewDimensions.assign(KArray);
            SDVariable KNewDimensions = sd.var(KArrayNewDimensionsPopulated);

            INDArray VArray = vLinear.getArr();
            INDArray VArrayNewDimensions = Nd4j.create(batchSize, V.eval().shape()[2], numHeads, newEmbSize);
            INDArray VArrayNewDimensionsPopulated = VArrayNewDimensions.assign(VArray);
            SDVariable VNewDimensions = sd.var(VArrayNewDimensionsPopulated);

            System.out.println(" MultiHeadAttention - forward - Arrays.toString(QNewDimensions.getShape()) - "+ Arrays.toString(QNewDimensions.getShape()));
            System.out.println(" MultiHeadAttention - forward - QNewDimensions.eval().shapeInfoToString() - "+ QNewDimensions.eval().shapeInfoToString());
            System.out.println(" MultiHeadAttention - forward - Arrays.toString(KNewDimensions.getShape()) - "+ Arrays.toString(KNewDimensions.getShape()));
            System.out.println(" MultiHeadAttention - forward - KNewDimensions.eval().shapeInfoToString() - "+ KNewDimensions.eval().shapeInfoToString());
            System.out.println(" MultiHeadAttention - forward - Arrays.toString(VNewDimensions.getShape()) - "+ Arrays.toString(VNewDimensions.getShape()));
            System.out.println(" MultiHeadAttention - forward - VNewDimensions.eval().shapeInfoToString() - "+ VNewDimensions.eval().shapeInfoToString());

//        # permuting the dimensions of Q, K and V
//        Q = Q.permute(0,2,1,3)                                          # Shape = (B, H, N, C//H)
//        K = K.permute(0,2,1,3)                                          # Shape = (B, H, N, C//H)
//        V = V.permute(0,2,1,3)                                          # Shape = (B, H, N, C//H)

            SDVariable QNewDimensionsPermuted = sd.permute(QNewDimensions, 0,2,1,3);
            SDVariable KNewDimensionsPermuted = sd.permute(KNewDimensions, 0,2,1,3);
            SDVariable VNewDimensionsPermuted = sd.permute(VNewDimensions, 0,2,1,3);

            System.out.println(" MultiHeadAttention - forward - Arrays.toString(QNewDimensionsPermuted.getShape()) 2- "+ Arrays.toString(QNewDimensionsPermuted.getShape()));
            System.out.println(" MultiHeadAttention - forward - QNewDimensionsPermuted.eval().shapeInfoToString() 2- "+ QNewDimensionsPermuted.eval().shapeInfoToString());
            System.out.println(" MultiHeadAttention - forward - Arrays.toString(KNewDimensionsPermuted.getShape()) 2- "+ Arrays.toString(KNewDimensionsPermuted.getShape()));
            System.out.println(" MultiHeadAttention - forward - KNewDimensionsPermuted.eval().shapeInfoToString() 2- "+ KNewDimensionsPermuted.eval().shapeInfoToString());
            System.out.println(" MultiHeadAttention - forward - Arrays.toString(VNewDimensionsPermuted.getShape()) 2- "+ Arrays.toString(VNewDimensionsPermuted.getShape()));
            System.out.println(" MultiHeadAttention - forward - VNewDimensionsPermuted.eval().shapeInfoToString() 2- "+ VNewDimensionsPermuted.eval().shapeInfoToString());

//        # calculating attention
//        attn_output = attention(Q, K, V, mask, self.dropout)            # Shape = (B, H, N, C//H)

            SDVariable attnOutput = attention(QNewDimensionsPermuted, KNewDimensionsPermuted, VNewDimensionsPermuted, mask, dropout);
            System.out.println(" MultiHeadAttention - forward - Arrays.toString(attnOutput.getShape()) 0- "+ Arrays.toString(attnOutput.getShape()));
            System.out.println(" MultiHeadAttention - forward - attnOutput.eval().shapeInfoToString() 0- "+ attnOutput.eval().shapeInfoToString());
            System.out.println(" MultiHeadAttention - forward - attnOutput.eval() 0- "+ attnOutput.eval());


//        # permuting the dimensions of attn_output and collapsing
//        # the num_heads dimension
//        attn_output = attn_output.permute(0,2,1,3)                      # Shape = (B, N, H, C//H)
//        attn_output = attn_output.reshape(batch_size, -1, self.emb_size)# Shape = (B, N, C)

            attnOutput = attnOutput.permute(0, 2, 1, 3);
            System.out.println(" MultiHeadAttention - forward - Arrays.toString(attnOutput.getShape()) permuted- "+ Arrays.toString(attnOutput.getShape()));
            System.out.println(" MultiHeadAttention - forward - attnOutput.eval().shapeInfoToString() permuted- "+ attnOutput.eval().shapeInfoToString());
            attnOutput = attnOutput.reshape(batchSize, -1, embSize);
            System.out.println(" MultiHeadAttention - forward - Arrays.toString(attnOutput.getShape()) 1- "+ Arrays.toString(attnOutput.getShape()));
            System.out.println(" MultiHeadAttention - forward - attnOutput.eval().shapeInfoToString() 1- "+ attnOutput.eval().shapeInfoToString());
            System.out.println(" MultiHeadAttention - forward - attnOutput.eval() 1- "+ attnOutput.eval());

            // applying linear layer to the output of the attention layer
            attnOutput = sd.mmul(attnOutput, postAttWeights);
            System.out.println(" MultiHeadAttention - forward - Arrays.toString(attnOutput.getShape()) mmul weights- "+ Arrays.toString(attnOutput.getShape()));
            System.out.println(" MultiHeadAttention - forward - attnOutput.eval().shapeInfoToString() mmul weights- "+ attnOutput.eval().shapeInfoToString());
            attnOutput = attnOutput.add(postAttBias);
            System.out.println(" MultiHeadAttention - forward - Arrays.toString(attnOutput.getShape()) add biaa- "+ Arrays.toString(attnOutput.getShape()));
            System.out.println(" MultiHeadAttention - forward - attnOutput.eval().shapeInfoToString() add bias- "+ attnOutput.eval().shapeInfoToString());
//            attnOutput = sd.mmul(attnOutput, sd.getVariable("postAttWeights"));
//            attnOutput = attnOutput.add(sd.getVariable("postAttBias"));

//        # applying linear layer to output of attention layer
//        attn_output = self.post_att(attn_output)                        # Shape = (B, N, C)

            SDVariable weights5 = sd.var("weights5"+" - "+mRandomNumericalId, new XavierInitScheme('c', encoder_ip_size, model_op_size), DataType.FLOAT, attnOutput.eval().shape()[0], attnOutput.eval().shape()[1], attnOutput.eval().shape()[2]);

            SDVariable attnOutputFinal = linear(attnOutput, weights5, bias);
//            attnOutput = sd.mmul(attnOutput, postAttWeights2);
//            attnOutput = attnOutput.add(postAttBias2);

//        return attn_output

            return attnOutputFinal;
        }

        public SDVariable attention(SDVariable Q, SDVariable K, SDVariable V, SDVariable mask, double dropout) {
//            public INDArray attention(INDArray Q, INDArray K, INDArray V, INDArray mask, Double dropout) {
//            public static INDArray attention(INDArray Q, INDArray K, INDArray V, INDArray mask, Double dropout) {

            System.out.println(" MultiHeadAttention - attention - Arrays.toString(Q.getShape()) - "+ Arrays.toString(Q.getShape()));
            System.out.println(" MultiHeadAttention - attention - Q.eval().shapeInfoToString() - "+ Q.eval().shapeInfoToString());
            System.out.println(" MultiHeadAttention - attention - Arrays.toString(K.getShape()) - "+ Arrays.toString(K.getShape()));
            System.out.println(" MultiHeadAttention - attention - K.eval().shapeInfoToString() - "+ K.eval().shapeInfoToString());

            // finding the embedding size
            long newEmbSize = Q.getArr().shape()[0];
//            long newEmbSize = Q.eval().shape()[0];

            // calculating attention scores
//            SDVariable Qpermuted = sd.permute(Q, 0, 3, 2, 1);
//            System.out.println(" MultiHeadAttention - attention - Arrays.toString(Qpermuted.getShape()) 2- "+ Arrays.toString(Qpermuted.getShape()));
//            System.out.println(" MultiHeadAttention - attention - Qpermuted.eval().shapeInfoToString() 2- "+ Qpermuted.eval().shapeInfoToString());

            SDVariable Kpermuted = sd.permute(Q, 0, 1, 3, 2);
            System.out.println(" MultiHeadAttention - attention - Arrays.toString(Kpermuted.getShape()) 2- "+ Arrays.toString(Kpermuted.getShape()));
            System.out.println(" MultiHeadAttention - attention - Kpermuted.eval().shapeInfoToString() 2- "+ Kpermuted.eval().shapeInfoToString());

            // calculating attention scores
            SDVariable scores = Q.mmul(Kpermuted).div(Math.sqrt(newEmbSize));
//            SDVariable scores = Q.mmul(sd.transpose(K)).div(Math.sqrt(newEmbSize));
//            SDVariable scores = Qpermuted.mmul(sd.transpose(K)).div(Math.sqrt(newEmbSize));
//            INDArray scores = Q.mmul(K.transpose()).div(Math.sqrt(newEmbSize));
            System.out.println(" MultiHeadAttention - attention - Arrays.toString(sd.transpose(K).getShape()) 2- "+ Arrays.toString(sd.transpose(K).getShape()));
            System.out.println(" MultiHeadAttention - attention - sd.transpose(K).eval().shapeInfoToString() 2- "+ sd.transpose(K).eval().shapeInfoToString());
            System.out.println(" MultiHeadAttention - attention - Arrays.toString(scores.getShape()) 2- "+ Arrays.toString(scores.getShape()));
            System.out.println(" MultiHeadAttention - attention - scores.eval().shapeInfoToString() 2- "+ scores.eval().shapeInfoToString());
            System.out.println(" MultiHeadAttention - attention - scores.eval() - "+ scores.eval());

            // applying mask on the attention
            if (mask != null) {
                sd.matchCondition(scores, Conditions.equals(0.0));
//                BooleanIndexing.replaceWhere(scores, -1e9, Conditions.equals(0.0));
//                    scores = scores.masked_fill_(mask == 0, -1e9)
            }

            SDVariable pAttn = new SDVariable();
            // applying softmax layer and calculating probability of attention
            if (dropout <= 0.0) {
                pAttn = sd.nn.softmax(scores, 2);
//            INDArray pAttn = Transforms.softmax(scores, 2);
            } else {
                // applying dropout
//              p_attn = dropout(p_attn)
                pAttn = dropout(sd.nn.softmax(scores, 2), dropout);

            }
            System.out.println(" MultiHeadAttention - attention - Arrays.toString(pAttn.getShape()) 2- "+ Arrays.toString(pAttn.getShape()));
            System.out.println(" MultiHeadAttention - attention - pAttn.eval().shapeInfoToString() 2- "+ pAttn.eval().shapeInfoToString());
            System.out.println(" MultiHeadAttention - attention - pAttn.eval() 2- "+ pAttn.eval());

            // multiplying the probability of attention with Values (V)
            SDVariable attnOutput = pAttn.mmul(V);
//            INDArray attnOutput = pAttn.mmul(V);
            System.out.println(" MultiHeadAttention - attention - Arrays.toString(attnOutput.getShape()) 2- "+ Arrays.toString(attnOutput.getShape()));
            System.out.println(" MultiHeadAttention - attention - attnOutput.eval().shapeInfoToString() 2- "+ attnOutput.eval().shapeInfoToString());
            System.out.println(" MultiHeadAttention - attention - attnOutput.eval() 2- "+ attnOutput.eval());

            return attnOutput;
        }

//        private Pair<SDVariable, SDVariable> attention(SDVariable query, SDVariable key, SDVariable value, SDVariable mask, double dropout) {
////        private Pair<INDArray, INDArray> attention(INDArray query, INDArray key, INDArray value, INDArray mask, double dropout) {
//
//            SDVariable scores = sd.mmul(query, key.permute(0, 2, 1)).div(Math.sqrt(embSize));
////            SDVariable scores = sd.mmul(query, key.permute(0, 2, 1)).div(Math.sqrt(dModel));
////        INDArray scores = Nd4j.matmul(query, key.transpose(-2, -1)).div(Math.sqrt(dModel));
//            if (mask != null) {
//                INDArray scoresArray = scores.getArr();
//                applyMask(scoresArray, mask.getArr());
//                sd.associateArrayWithVariable(scoresArray, scores);
////            scores.setArray(scoresArray);
////            scores = scores.(mask.eq(0), -1e9);
////            scores = scores.maskedFill(mask.eq(0), -1e9);
//            }
//            SDVariable pAttn = sd.nn.softmax(scores, -1);
////        INDArray pAttn = Softmax.softmax(scores, -1);
//            if (dropout > 0.0) {
//                pAttn = dropout(pAttn, dropout);
////            pAttn = Dropout.dropout(pAttn, dropout);
//            }
//            SDVariable x = sd.mmul(pAttn, value);
//            return new Pair<>(x, pAttn);
//        }

        public static INDArray[] scaledDotProduct(INDArray q, INDArray k, INDArray v, INDArray mask) {
            long d_k = q.size(q.rank() - 1);
            INDArray attnLogits = q.mmul(k.permute(-2, -1)).div(Math.sqrt(d_k));
            if (mask != null) {
                applyMask(attnLogits, mask);
//            attnLogits = attnLogits.put(mask.eq(0), Nd4j.ones(mask.shape()).mul(-9e15));
//            attnLogits = attnLogits.putMaskedArray(mask.eq(0), Nd4j.ones(mask.shape()).mul(-9e15));
            }
            INDArray attention = Transforms.softmax(attnLogits);
//        INDArray attention = Transforms.softmax(attnLogits, 1);
            INDArray values = attention.mmul(v);
            INDArray[] scaledDotProductResult = null;
            scaledDotProductResult[0] = attention;
            scaledDotProductResult[1] = values;
            return scaledDotProductResult;
        }

        public static void applyMask(INDArray to, INDArray mask) {
            //Two possibilities exist: it's *per example* masking, or it's *per output* masking
            //These cases have different mask shapes. Per example: column vector. Per output: same shape as score array
            if (mask.isColumnVectorOrScalar()) {
                to.muliColumnVector(mask);
            } else if (Arrays.equals(to.shape(), mask.shape())) {
                to.muli(mask);
            } else {
                throw new IllegalStateException("Invalid mask array: per-example masking should be a column vector, "
                        + "per output masking arrays should be the same shape as the labels array. Mask shape: "
                        + Arrays.toString(mask.shape()) + ", output shape: " + Arrays.toString(to.shape()));
            }
        }


    }


    // Multi Head Attention Layer
    public static class MultiHeadAttention2 {

//    Class to create the multi head attention layer for
//    encoder and decoder
//
//        Class constructor
//
//        INPUT:
//        num_head - (int) number of heads in multi head attention layer
//        emb_size - (int) embedding size of the data
//        dropout - (float) dropout percentage. Default value = 0.1

//        private int embSize;
//        private int numHeads;
//        private double dropout;

//        private SameDiff sd;

        SDVariable qLinearWeights2 = new SDVariable();
        SDVariable kLinearWeights2 = new SDVariable();
        SDVariable vLinearWeights2 = new SDVariable();
        SDVariable postAttWeights2 = new SDVariable();
        SDVariable postAttBias2 = new SDVariable();
        SDVariable dropout2 = new SDVariable();


        public MultiHeadAttention2(int numHeads1, int embSize1, double dropout1) {

            embSize = embSize1;
            numHeads = numHeads1;
            dropout = dropout1;
//            this.embSize = embSize;
//            this.numHeads = numHeads;
//            this.dropout = dropout;

            System.out.println("embSize: " + embSize);
            System.out.println("numHeads: " + numHeads);

            // making sure that the embedding size is divisible by the number of heads
            if (embSize % numHeads != 0) {
                throw new IllegalArgumentException("Embedding size must be divisible by the number of heads.");
            }

//            INDArray arr = Nd4j.create(embSize,embSize);
//            SDVariable input = sd.var("input", arr);
//
//            SDVariable q_linear = linear(input, weights, bias);
//            SDVariable k_linear = linear(input, weights, bias);
//            SDVariable v_linear = linear(input, weights, bias);
//
//            SDVariable post_att = linear(input, weights, bias);
//
//            SDVariable dropoutSDV = dropout(input, dropout);

        }
        public SDVariable forward(SameDiff sd, SDVariable Q, SDVariable K, SDVariable V, SDVariable mask) {

//           forward function for MultiHeadAttention
//
//           INPUT:
//           Q - (torch tensor) query for the transformer. Shape = (B, N, C)
//           K - (torch tensor) keys for the transformer. Shape = (B, N, C)
//           V - (torch tensor) values for the transformer. Shape = (B, N, C)
//           mask - (torch tensor) mask for decoder multi head attention layer
//
//           OUTPUT:
//           att_output - (torch tensor) output of the multi head attention layer. Shape = (B, N, C)

            Random mRandom = new Random();
            int mRandomNumericalId = mRandom.nextInt(100000);

            System.out.println(" MultiHeadAttention2 - forward - Arrays.toString(Q.getShape()) - "+ Arrays.toString(Q.getShape()));
            System.out.println(" MultiHeadAttention2 - forward - Q.eval().shapeInfoToString() - "+ Q.eval().shapeInfoToString());

            // creating MLP layer for post-attention
            postAttWeights2 = sd.var(Nd4j.randn(batch_size, embSize, embSize));
            postAttBias2 = sd.var(Nd4j.zeros(1, embSize));

            // creating dropout layer
            dropout2 = sd.var(Nd4j.scalar(dropout));

            // Same mask applied to all h heads.
            if (mask != null) {
                sd.expandDims(mask, 1);
            }


//        # passing the Q, K, and V through 1 layer MLP
//        Q, K, V = self.q_linear(Q), self.k_linear(K), self.v_linear(V)  # Shape = (B, N, C)

            SDVariable weights2 = sd.var("weights2"+" - "+mRandomNumericalId, new XavierInitScheme('c', encoder_ip_size, model_op_size), DataType.FLOAT, Q.eval().shape()[0], Q.eval().shape()[1], Q.eval().shape()[2]);

            SDVariable qLinear = linear(Q, weights2, bias);
            SDVariable kLinear = linear(K, weights2, bias);
            SDVariable vLinear = linear(V, weights2, bias);

//        # splitting Q, K and V based on num_heads
//        batch_size = Q.shape[0]
//        new_emb_size = self.emb_size // self.num_heads

            int batchSize = (int) Q.eval().shape()[0];
            int newEmbSize = embSize / numHeads;

//        Q = Q.reshape(batch_size, -1, self.num_heads, new_emb_size)     # Shape = (B, N, H, C//H)
//        K = K.reshape(batch_size, -1, self.num_heads, new_emb_size)     # Shape = (B, N, H, C//H)
//        V = V.reshape(batch_size, -1, self.num_heads, new_emb_size)     # Shape = (B, N, H, C//H)

            INDArray QArray = qLinear.getArr();
//            INDArray QArrayNewDimensions = Nd4j.create(batchSize, Q.getShape()[2], numHeads, newEmbSize);
            INDArray QArrayNewDimensions = Nd4j.create(batchSize, Q.eval().shape()[2], numHeads, newEmbSize);
            INDArray QArrayNewDimensionsPopulated = QArrayNewDimensions.assign(QArray);
            SDVariable QNewDimensions = sd.var(QArrayNewDimensionsPopulated);

            INDArray KArray = kLinear.getArr();
            INDArray KArrayNewDimensions = Nd4j.create(batchSize, K.eval().shape()[2], numHeads, newEmbSize);
            INDArray KArrayNewDimensionsPopulated = KArrayNewDimensions.assign(KArray);
            SDVariable KNewDimensions = sd.var(KArrayNewDimensionsPopulated);

            INDArray VArray = vLinear.getArr();
            INDArray VArrayNewDimensions = Nd4j.create(batchSize, V.eval().shape()[2], numHeads, newEmbSize);
            INDArray VArrayNewDimensionsPopulated = VArrayNewDimensions.assign(VArray);
            SDVariable VNewDimensions = sd.var(VArrayNewDimensionsPopulated);

            System.out.println(" MultiHeadAttention2 - forward - Arrays.toString(QNewDimensions.getShape()) - "+ Arrays.toString(QNewDimensions.getShape()));
            System.out.println(" MultiHeadAttention2 - forward - QNewDimensions.eval().shapeInfoToString() - "+ QNewDimensions.eval().shapeInfoToString());
            System.out.println(" MultiHeadAttention2 - forward - Arrays.toString(KNewDimensions.getShape()) - "+ Arrays.toString(KNewDimensions.getShape()));
            System.out.println(" MultiHeadAttention2 - forward - KNewDimensions.eval().shapeInfoToString() - "+ KNewDimensions.eval().shapeInfoToString());
            System.out.println(" MultiHeadAttention2 - forward - Arrays.toString(VNewDimensions.getShape()) - "+ Arrays.toString(VNewDimensions.getShape()));
            System.out.println(" MultiHeadAttention2 - forward - VNewDimensions.eval().shapeInfoToString() - "+ VNewDimensions.eval().shapeInfoToString());

//        # permuting the dimensions of Q, K and V
//        Q = Q.permute(0,2,1,3)                                          # Shape = (B, H, N, C//H)
//        K = K.permute(0,2,1,3)                                          # Shape = (B, H, N, C//H)
//        V = V.permute(0,2,1,3)                                          # Shape = (B, H, N, C//H)

            SDVariable QNewDimensionsPermuted = sd.permute(QNewDimensions, 0,2,1,3);
            SDVariable KNewDimensionsPermuted = sd.permute(KNewDimensions, 0,2,1,3);
            SDVariable VNewDimensionsPermuted = sd.permute(VNewDimensions, 0,2,1,3);

            System.out.println(" MultiHeadAttention2 - forward - Arrays.toString(QNewDimensionsPermuted.getShape()) 2- "+ Arrays.toString(QNewDimensionsPermuted.getShape()));
            System.out.println(" MultiHeadAttention2 - forward - QNewDimensionsPermuted.eval().shapeInfoToString() 2- "+ QNewDimensionsPermuted.eval().shapeInfoToString());
            System.out.println(" MultiHeadAttention2 - forward - Arrays.toString(KNewDimensionsPermuted.getShape()) 2- "+ Arrays.toString(KNewDimensionsPermuted.getShape()));
            System.out.println(" MultiHeadAttention2 - forward - KNewDimensionsPermuted.eval().shapeInfoToString() 2- "+ KNewDimensionsPermuted.eval().shapeInfoToString());
            System.out.println(" MultiHeadAttention2 - forward - Arrays.toString(VNewDimensionsPermuted.getShape()) 2- "+ Arrays.toString(VNewDimensionsPermuted.getShape()));
            System.out.println(" MultiHeadAttention2 - forward - VNewDimensionsPermuted.eval().shapeInfoToString() 2- "+ VNewDimensionsPermuted.eval().shapeInfoToString());

//        # calculating attention
//        attn_output = attention(Q, K, V, mask, self.dropout)            # Shape = (B, H, N, C//H)

            SDVariable attnOutput = attention(QNewDimensionsPermuted, KNewDimensionsPermuted, VNewDimensionsPermuted, mask, dropout);
            System.out.println(" MultiHeadAttention2 - forward - Arrays.toString(attnOutput.getShape()) 0- "+ Arrays.toString(attnOutput.getShape()));
            System.out.println(" MultiHeadAttention2 - forward - attnOutput.eval().shapeInfoToString() 0- "+ attnOutput.eval().shapeInfoToString());
            System.out.println(" MultiHeadAttention2 - forward - attnOutput.eval() 0- "+ attnOutput.eval());


//        # permuting the dimensions of attn_output and collapsing
//        # the num_heads dimension
//        attn_output = attn_output.permute(0,2,1,3)                      # Shape = (B, N, H, C//H)
//        attn_output = attn_output.reshape(batch_size, -1, self.emb_size)# Shape = (B, N, C)

            attnOutput = attnOutput.permute(0, 2, 1, 3);
            attnOutput = attnOutput.reshape(batchSize, -1, embSize);
            System.out.println(" MultiHeadAttention2 - forward - Arrays.toString(attnOutput.getShape()) 1- "+ Arrays.toString(attnOutput.getShape()));
            System.out.println(" MultiHeadAttention2 - forward - attnOutput.eval().shapeInfoToString() 1- "+ attnOutput.eval().shapeInfoToString());
            System.out.println(" MultiHeadAttention2 - forward - attnOutput.eval() 1- "+ attnOutput.eval());


//        # applying linear layer to output of attention layer
//        attn_output = self.post_att(attn_output)                        # Shape = (B, N, C)

            SDVariable weights3 = sd.var("weights3"+" - "+mRandomNumericalId, new XavierInitScheme('c', encoder_ip_size, model_op_size), DataType.FLOAT, attnOutput.eval().shape()[0], attnOutput.eval().shape()[1], attnOutput.eval().shape()[2]);

            SDVariable attnOutputFinal = linear(attnOutput, weights3, bias);
//            attnOutput = sd.mmul(attnOutput, postAttWeights2);
//            attnOutput = attnOutput.add(postAttBias2);

//        return attn_output

            return attnOutputFinal;
        }
        public SDVariable attention(SDVariable Q, SDVariable K, SDVariable V, SDVariable mask, double dropout) {
//            public INDArray attention(INDArray Q, INDArray K, INDArray V, INDArray mask, Double dropout) {
//            public static INDArray attention(INDArray Q, INDArray K, INDArray V, INDArray mask, Double dropout) {

            System.out.println(" MultiHeadAttention2 - attention - Arrays.toString(Q.getShape()) - "+ Arrays.toString(Q.getShape()));
            System.out.println(" MultiHeadAttention2 - attention - Q.eval().shapeInfoToString() - "+ Q.eval().shapeInfoToString());
            System.out.println(" MultiHeadAttention2 - attention - Arrays.toString(K.getShape()) - "+ Arrays.toString(K.getShape()));
            System.out.println(" MultiHeadAttention2 - attention - K.eval().shapeInfoToString() - "+ K.eval().shapeInfoToString());

            // finding the embedding size
            long newEmbSize = Q.getArr().shape()[0];
//            long newEmbSize = Q.eval().shape()[0];
            System.out.println(" MultiHeadAttention2 - attention - newEmbSize - "+ newEmbSize);

            // calculating attention scores
//            SDVariable Qpermuted = sd.permute(Q, 0, 3, 2, 1);
//            System.out.println(" MultiHeadAttention2 - attention - Arrays.toString(Qpermuted.getShape()) 2- "+ Arrays.toString(Qpermuted.getShape()));
//            System.out.println(" MultiHeadAttention2 - attention - Qpermuted.eval().shapeInfoToString() 2- "+ Qpermuted.eval().shapeInfoToString());

            SDVariable Kpermuted = sd.permute(Q, 0, 1, 3, 2);
            System.out.println(" MultiHeadAttention2 - attention - Arrays.toString(Kpermuted.getShape()) 2- "+ Arrays.toString(Kpermuted.getShape()));
            System.out.println(" MultiHeadAttention2 - attention - Kpermuted.eval().shapeInfoToString() 2- "+ Kpermuted.eval().shapeInfoToString());

            SDVariable scores = Q.mmul(Kpermuted).div(Math.sqrt(newEmbSize));
//            SDVariable scores = Q.mmul(sd.transpose(K)).div(Math.sqrt(newEmbSize));
//            SDVariable scores = Qpermuted.mmul(sd.transpose(K)).div(Math.sqrt(newEmbSize));
//            INDArray scores = Q.mmul(K.transpose()).div(Math.sqrt(newEmbSize));
            System.out.println(" MultiHeadAttention2 - attention - Arrays.toString(sd.transpose(K).getShape()) 2- "+ Arrays.toString(sd.transpose(K).getShape()));
            System.out.println(" MultiHeadAttention2 - attention - sd.transpose(K).eval().shapeInfoToString() 2- "+ sd.transpose(K).eval().shapeInfoToString());
            System.out.println(" MultiHeadAttention2 - attention - Arrays.toString(scores.getShape()) 2- "+ Arrays.toString(scores.getShape()));
            System.out.println(" MultiHeadAttention2 - attention - scores.eval().shapeInfoToString() 2- "+ scores.eval().shapeInfoToString());
            System.out.println(" MultiHeadAttention2 - attention - scores.eval() - "+ scores.eval());

            // applying mask on the attention
            if (mask != null) {
                sd.matchCondition(scores, Conditions.equals(0.0));
//                BooleanIndexing.replaceWhere(scores, -1e9, Conditions.equals(0.0));
//                    scores = scores.masked_fill_(mask == 0, -1e9)
            }

            SDVariable pAttn = new SDVariable();
            // applying softmax layer and calculating probability of attention
            if (dropout <= 0.0) {
                pAttn = sd.nn.softmax(scores, 2);
//            INDArray pAttn = Transforms.softmax(scores, 2);
            } else {
                // applying dropout
//              p_attn = dropout(p_attn)
                pAttn = dropout(sd.nn.softmax(scores, 2), dropout);

            }
            System.out.println(" MultiHeadAttention2 - attention - Arrays.toString(pAttn.getShape()) 2- "+ Arrays.toString(pAttn.getShape()));
            System.out.println(" MultiHeadAttention2 - attention - pAttn.eval().shapeInfoToString() 2- "+ pAttn.eval().shapeInfoToString());
            System.out.println(" MultiHeadAttention2 - attention - pAttn.eval() 2- "+ pAttn.eval());

            // multiplying the probability of attention with Values (V)
            SDVariable attnOutput = pAttn.mmul(V);
//            INDArray attnOutput = pAttn.mmul(V);
            System.out.println(" MultiHeadAttention2 - attention - Arrays.toString(attnOutput.getShape()) 2- "+ Arrays.toString(attnOutput.getShape()));
            System.out.println(" MultiHeadAttention2 - attention - attnOutput.eval().shapeInfoToString() 2- "+ attnOutput.eval().shapeInfoToString());
            System.out.println(" MultiHeadAttention2 - attention - attnOutput.eval() 2- "+ attnOutput.eval());

            return attnOutput;
        }


//        private Pair<SDVariable, SDVariable> attention(SDVariable query, SDVariable key, SDVariable value, SDVariable mask, double dropout) {
////        private Pair<INDArray, INDArray> attention(INDArray query, INDArray key, INDArray value, INDArray mask, double dropout) {
//
//            SDVariable scores = sd.mmul(query, key.permute(0, 2, 1)).div(Math.sqrt(embSize));
////            SDVariable scores = sd.mmul(query, key.permute(0, 2, 1)).div(Math.sqrt(dModel));
////        INDArray scores = Nd4j.matmul(query, key.transpose(-2, -1)).div(Math.sqrt(dModel));
//            if (mask != null) {
//                INDArray scoresArray = scores.getArr();
//                applyMask(scoresArray, mask.getArr());
//                sd.associateArrayWithVariable(scoresArray, scores);
////            scores.setArray(scoresArray);
////            scores = scores.(mask.eq(0), -1e9);
////            scores = scores.maskedFill(mask.eq(0), -1e9);
//            }
//            SDVariable pAttn = sd.nn.softmax(scores, -1);
////        INDArray pAttn = Softmax.softmax(scores, -1);
//            if (dropout > 0.0) {
//                pAttn = dropout(pAttn, dropout);
////            pAttn = Dropout.dropout(pAttn, dropout);
//            }
//            SDVariable x = sd.mmul(pAttn, value);
//            return new Pair<>(x, pAttn);
//        }

        public static INDArray[] scaledDotProduct(INDArray q, INDArray k, INDArray v, INDArray mask) {
            long d_k = q.size(q.rank() - 1);
            INDArray attnLogits = q.mmul(k.permute(-2, -1)).div(Math.sqrt(d_k));
            if (mask != null) {
                applyMask(attnLogits, mask);
//            attnLogits = attnLogits.put(mask.eq(0), Nd4j.ones(mask.shape()).mul(-9e15));
//            attnLogits = attnLogits.putMaskedArray(mask.eq(0), Nd4j.ones(mask.shape()).mul(-9e15));
            }
            INDArray attention = Transforms.softmax(attnLogits);
//        INDArray attention = Transforms.softmax(attnLogits, 1);
            INDArray values = attention.mmul(v);
            INDArray[] scaledDotProductResult = null;
            scaledDotProductResult[0] = attention;
            scaledDotProductResult[1] = values;
            return scaledDotProductResult;
        }

        public static void applyMask(INDArray to, INDArray mask) {
            //Two possibilities exist: it's *per example* masking, or it's *per output* masking
            //These cases have different mask shapes. Per example: column vector. Per output: same shape as score array
            if (mask.isColumnVectorOrScalar()) {
                to.muliColumnVector(mask);
            } else if (Arrays.equals(to.shape(), mask.shape())) {
                to.muli(mask);
            } else {
                throw new IllegalStateException("Invalid mask array: per-example masking should be a column vector, "
                        + "per output masking arrays should be the same shape as the labels array. Mask shape: "
                        + Arrays.toString(mask.shape()) + ", output shape: " + Arrays.toString(to.shape()));
            }
        }


    }


    // Multi Head Attention Layer
    public static class MultiHeadAttention3 {

//    Class to create the multi head attention layer for
//    encoder and decoder
//
//        Class constructor
//
//        INPUT:
//        num_head - (int) number of heads in multi head attention layer
//        emb_size - (int) embedding size of the data
//        dropout - (float) dropout percentage. Default value = 0.1

//        private int embSize;
//        private int numHeads;
//        private double dropout;

//        private SameDiff sd;

        SDVariable qLinearWeights3 = new SDVariable();
        SDVariable kLinearWeights3 = new SDVariable();
        SDVariable vLinearWeights3 = new SDVariable();
        SDVariable postAttWeights3 = new SDVariable();
        SDVariable postAttBias3 = new SDVariable();
        SDVariable dropout3 = new SDVariable();


        public MultiHeadAttention3(int numHeads1, int embSize1, double dropout1) {

            embSize = embSize1;
            numHeads = numHeads1;
            dropout = dropout1;
//            this.embSize = embSize;
//            this.numHeads = numHeads;
//            this.dropout = dropout;

            System.out.println("embSize: " + embSize);
            System.out.println("numHeads: " + numHeads);

            // making sure that the embedding size is divisible by the number of heads
            if (embSize % numHeads != 0) {
                throw new IllegalArgumentException("Embedding size must be divisible by the number of heads.");
            }

//            INDArray arr = Nd4j.create(embSize,embSize);
//            SDVariable input = sd.var("input", arr);
//
//            SDVariable q_linear = linear(input, weights, bias);
//            SDVariable k_linear = linear(input, weights, bias);
//            SDVariable v_linear = linear(input, weights, bias);
//
//            SDVariable post_att = linear(input, weights, bias);
//
//            SDVariable dropoutSDV = dropout(input, dropout);

        }

        public SDVariable forward(SameDiff sd, SDVariable Q, SDVariable K, SDVariable V, SDVariable mask) {
//            public INDArray forward(INDArray Q, INDArray K, INDArray V, INDArray mask) {

//           forward function for MultiHeadAttention
//
//           INPUT:
//           Q - (torch tensor) query for the transformer. Shape = (B, N, C)
//           K - (torch tensor) keys for the transformer. Shape = (B, N, C)
//           V - (torch tensor) values for the transformer. Shape = (B, N, C)
//           mask - (torch tensor) mask for decoder multi head attention layer
//
//           OUTPUT:
//           att_output - (torch tensor) output of the multi head attention layer. Shape = (B, N, C)

            mRandom = new Random();
            mRandomNumericalId = mRandom.nextInt(100000);

            System.out.println(" MultiHeadAttention3 - forward - Arrays.toString(Q.getShape()) - "+ Arrays.toString(Q.getShape()));
            System.out.println(" MultiHeadAttention3 - forward - Q.eval().shapeInfoToString() - "+ Q.eval().shapeInfoToString());

            // creating MLP layer for post-attention
            postAttWeights3 = sd.var(Nd4j.randn(batch_size, embSize, embSize));
            postAttBias3 = sd.var(Nd4j.zeros(1, embSize));
//            postAttBias3 = sd.var("postAttBias3"+" - "+mRandomNumericalId, Nd4j.zeros(1, embSize));

            // creating dropout layer
            dropout3 = sd.var(Nd4j.scalar(dropout));

            // Same mask applied to all h heads.
            if (mask != null) {
                sd.expandDims(mask, 1);
//                mask = mask.reshape(mask.size(0), 1, mask.size(1), mask.size(2));
            }

//            qLinearWeights3 = sd.expandDims(qLinearWeights3, 2);
//            kLinearWeights3 = sd.expandDims(kLinearWeights3, 2);
//            vLinearWeights3 = sd.expandDims(vLinearWeights3, 2);

//        # passing the Q, K, and V through 1 layer MLP
//        Q, K, V = self.q_linear(Q), self.k_linear(K), self.v_linear(V)  # Shape = (B, N, C)

            SDVariable weghts6 = sd.var("weghts6"+" - "+mRandomNumericalId, new XavierInitScheme('c', encoder_ip_size, model_op_size), DataType.FLOAT, Q.eval().shape()[0], Q.eval().shape()[1], Q.eval().shape()[2]);

            SDVariable qLinear = linear(Q, weghts6, bias);
            SDVariable kLinear = linear(K, weghts6, bias);
            SDVariable vLinear = linear(V, weghts6, bias);

//        # splitting Q, K and V based on num_heads
//        batch_size = Q.shape[0]
//        new_emb_size = self.emb_size // self.num_heads

            // passing the Q, K, and V through 1 layer MLP

            // splitting Q, K, and V based on num_heads
            int batchSize = (int) Q.eval().shape()[0];
//            int batchSize = (int) Q.getArr().shape()[0];
//            int batchSize = Q.size(0);
            int newEmbSize = embSize / numHeads;
            System.out.println(" MultiHeadAttention3 - forward - batchSize - "+ batchSize);
            System.out.println(" MultiHeadAttention3 - forward - newEmbSize - "+ newEmbSize);

//        Q = Q.reshape(batch_size, -1, self.num_heads, new_emb_size)     # Shape = (B, N, H, C//H)
//        K = K.reshape(batch_size, -1, self.num_heads, new_emb_size)     # Shape = (B, N, H, C//H)
//        V = V.reshape(batch_size, -1, self.num_heads, new_emb_size)     # Shape = (B, N, H, C//H)

            INDArray QArray = qLinear.getArr();
            INDArray QArrayNewDimensions = Nd4j.create(batchSize, Q.eval().shape()[2], numHeads, newEmbSize);
            INDArray QArrayNewDimensionsPopulated = QArrayNewDimensions.assign(QArray);
            SDVariable QNewDimensions = sd.var(QArrayNewDimensionsPopulated);

            INDArray KArray = kLinear.getArr();
            INDArray KArrayNewDimensions = Nd4j.create(batchSize, K.eval().shape()[2], numHeads, newEmbSize);
            INDArray KArrayNewDimensionsPopulated = KArrayNewDimensions.assign(KArray);
            SDVariable KNewDimensions = sd.var(KArrayNewDimensionsPopulated);

            INDArray VArray = vLinear.getArr();
            INDArray VArrayNewDimensions = Nd4j.create(batchSize, V.eval().shape()[2], numHeads, newEmbSize);
            INDArray VArrayNewDimensionsPopulated = VArrayNewDimensions.assign(VArray);
            SDVariable VNewDimensions = sd.var(VArrayNewDimensionsPopulated);

            System.out.println(" MultiHeadAttention3 - forward - Arrays.toString(QNewDimensions.getShape()) - "+ Arrays.toString(QNewDimensions.getShape()));
            System.out.println(" MultiHeadAttention3 - forward - QNewDimensions.eval().shapeInfoToString() - "+ QNewDimensions.eval().shapeInfoToString());
            System.out.println(" MultiHeadAttention3 - forward - Arrays.toString(KNewDimensions.getShape()) - "+ Arrays.toString(KNewDimensions.getShape()));
            System.out.println(" MultiHeadAttention3 - forward - KNewDimensions.eval().shapeInfoToString() - "+ KNewDimensions.eval().shapeInfoToString());
            System.out.println(" MultiHeadAttention3 - forward - Arrays.toString(VNewDimensions.getShape()) - "+ Arrays.toString(VNewDimensions.getShape()));
            System.out.println(" MultiHeadAttention3 - forward - VNewDimensions.eval().shapeInfoToString() - "+ VNewDimensions.eval().shapeInfoToString());

//        # permuting the dimensions of Q, K and V
//        Q = Q.permute(0,2,1,3)                                          # Shape = (B, H, N, C//H)
//        K = K.permute(0,2,1,3)                                          # Shape = (B, H, N, C//H)
//        V = V.permute(0,2,1,3)                                          # Shape = (B, H, N, C//H)

            SDVariable QNewDimensionsPermuted = sd.permute(QNewDimensions, 0,2,1,3);
            SDVariable KNewDimensionsPermuted = sd.permute(KNewDimensions, 0,2,1,3);
            SDVariable VNewDimensionsPermuted = sd.permute(VNewDimensions, 0,2,1,3);

            System.out.println(" MultiHeadAttention3 - forward - Arrays.toString(QNewDimensionsPermuted.getShape()) 2- "+ Arrays.toString(QNewDimensionsPermuted.getShape()));
            System.out.println(" MultiHeadAttention3 - forward - QNewDimensionsPermuted.eval().shapeInfoToString() 2- "+ QNewDimensionsPermuted.eval().shapeInfoToString());
            System.out.println(" MultiHeadAttention3 - forward - Arrays.toString(KNewDimensionsPermuted.getShape()) 2- "+ Arrays.toString(KNewDimensionsPermuted.getShape()));
            System.out.println(" MultiHeadAttention3 - forward - KNewDimensionsPermuted.eval().shapeInfoToString() 2- "+ KNewDimensionsPermuted.eval().shapeInfoToString());
            System.out.println(" MultiHeadAttention3 - forward - Arrays.toString(VNewDimensionsPermuted.getShape()) 2- "+ Arrays.toString(VNewDimensionsPermuted.getShape()));
            System.out.println(" MultiHeadAttention3 - forward - VNewDimensionsPermuted.eval().shapeInfoToString() 2- "+ VNewDimensionsPermuted.eval().shapeInfoToString());

//        # calculating attention
//        attn_output = attention(Q, K, V, mask, self.dropout)            # Shape = (B, H, N, C//H)

            SDVariable attnOutput = attention(QNewDimensionsPermuted, KNewDimensionsPermuted, VNewDimensionsPermuted, mask, dropout);
            System.out.println(" MultiHeadAttention3 - forward - Arrays.toString(attnOutput.getShape()) 0- "+ Arrays.toString(attnOutput.getShape()));
            System.out.println(" MultiHeadAttention3 - forward - attnOutput.eval().shapeInfoToString() 0- "+ attnOutput.eval().shapeInfoToString());
            System.out.println(" MultiHeadAttention3 - forward - attnOutput.eval() 0- "+ attnOutput.eval());


//        # permuting the dimensions of attn_output and collapsing
//        # the num_heads dimension
//        attn_output = attn_output.permute(0,2,1,3)                      # Shape = (B, N, H, C//H)
//        attn_output = attn_output.reshape(batch_size, -1, self.emb_size)# Shape = (B, N, C)

            attnOutput = attnOutput.permute(0, 2, 1, 3);
            attnOutput = attnOutput.reshape(batchSize, -1, embSize);
            System.out.println(" MultiHeadAttention3 - forward - Arrays.toString(attnOutput.getShape()) 1- "+ Arrays.toString(attnOutput.getShape()));
            System.out.println(" MultiHeadAttention3 - forward - attnOutput.eval().shapeInfoToString() 1- "+ attnOutput.eval().shapeInfoToString());
            System.out.println(" MultiHeadAttention3 - forward - attnOutput.eval() 1- "+ attnOutput.eval());


//        # applying linear layer to output of attention layer
//        attn_output = self.post_att(attn_output)                        # Shape = (B, N, C)

            SDVariable weights7 = sd.var("weights7"+" - "+mRandomNumericalId, new XavierInitScheme('c', encoder_ip_size, model_op_size), DataType.FLOAT, attnOutput.eval().shape()[0], attnOutput.eval().shape()[1], attnOutput.eval().shape()[2]);

            SDVariable attnOutputFinal = linear(attnOutput, weights7, bias);
//            attnOutput = sd.mmul(attnOutput, postAttWeights3);
//            attnOutput = attnOutput.add(postAttBias3);

//        return attn_output

            return attnOutputFinal;
        }

        public SDVariable attention(SDVariable Q, SDVariable K, SDVariable V, SDVariable mask, double dropout) {
//            public INDArray attention(INDArray Q, INDArray K, INDArray V, INDArray mask, Double dropout) {
//            public static INDArray attention(INDArray Q, INDArray K, INDArray V, INDArray mask, Double dropout) {

            System.out.println(" MultiHeadAttention3 - attention - Arrays.toString(Q.getShape()) - "+ Arrays.toString(Q.getShape()));
            System.out.println(" MultiHeadAttention3 - attention - Q.eval().shapeInfoToString() - "+ Q.eval().shapeInfoToString());
            System.out.println(" MultiHeadAttention3 - attention - Arrays.toString(K.getShape()) - "+ Arrays.toString(K.getShape()));
            System.out.println(" MultiHeadAttention3 - attention - K.eval().shapeInfoToString() - "+ K.eval().shapeInfoToString());

            // finding the embedding size
            long newEmbSize = Q.getArr().shape()[0];
//            long newEmbSize = Q.eval().shape()[0];
            System.out.println(" MultiHeadAttention3 - attention - newEmbSize - "+ newEmbSize);

            // calculating attention scores
//            SDVariable Qpermuted = sd.permute(Q, 0, 3, 2, 1);
//            System.out.println(" MultiHeadAttention3 - attention - Arrays.toString(Qpermuted.getShape()) 2- "+ Arrays.toString(Qpermuted.getShape()));
//            System.out.println(" MultiHeadAttention3 - attention - Qpermuted.eval().shapeInfoToString() 2- "+ Qpermuted.eval().shapeInfoToString());

            SDVariable Kpermuted = sd.permute(Q, 0, 1, 3, 2);
            System.out.println(" MultiHeadAttention3 - attention - Arrays.toString(Kpermuted.getShape()) 2- "+ Arrays.toString(Kpermuted.getShape()));
            System.out.println(" MultiHeadAttention3 - attention - Kpermuted.eval().shapeInfoToString() 2- "+ Kpermuted.eval().shapeInfoToString());

            SDVariable scores = Q.mmul(Kpermuted).div(Math.sqrt(newEmbSize));
//            SDVariable scores = Q.mmul(sd.transpose(K)).div(Math.sqrt(newEmbSize));
//            SDVariable scores = Qpermuted.mmul(sd.transpose(K)).div(Math.sqrt(newEmbSize));
//            INDArray scores = Q.mmul(K.transpose()).div(Math.sqrt(newEmbSize));
            System.out.println(" MultiHeadAttention3 - attention - Arrays.toString(sd.transpose(K).getShape()) 2- "+ Arrays.toString(sd.transpose(K).getShape()));
            System.out.println(" MultiHeadAttention3 - attention - sd.transpose(K).eval().shapeInfoToString() 2- "+ sd.transpose(K).eval().shapeInfoToString());
            System.out.println(" MultiHeadAttention3 - attention - Arrays.toString(scores.getShape()) 2- "+ Arrays.toString(scores.getShape()));
            System.out.println(" MultiHeadAttention3 - attention - scores.eval().shapeInfoToString() 2- "+ scores.eval().shapeInfoToString());
            System.out.println(" MultiHeadAttention3 - attention - scores.eval() - "+ scores.eval());

            // applying mask on the attention
            if (mask != null) {
                sd.matchCondition(scores, Conditions.equals(0.0));
//                BooleanIndexing.replaceWhere(scores, -1e9, Conditions.equals(0.0));
//                    scores = scores.masked_fill_(mask == 0, -1e9)
            }

            SDVariable pAttn = new SDVariable();
            // applying softmax layer and calculating probability of attention
            if (dropout <= 0.0) {
                pAttn = sd.nn.softmax(scores, 2);
//            INDArray pAttn = Transforms.softmax(scores, 2);
            } else {
                // applying dropout
//              p_attn = dropout(p_attn)
                pAttn = dropout(sd.nn.softmax(scores, 2), dropout);

            }
            System.out.println(" MultiHeadAttention3 - attention - Arrays.toString(pAttn.getShape()) 2- "+ Arrays.toString(pAttn.getShape()));
            System.out.println(" MultiHeadAttention3 - attention - pAttn.eval().shapeInfoToString() 2- "+ pAttn.eval().shapeInfoToString());
            System.out.println(" MultiHeadAttention3 - attention - pAttn.eval() 2- "+ pAttn.eval());

            // multiplying the probability of attention with Values (V)
            SDVariable attnOutput = pAttn.mmul(V);
//            INDArray attnOutput = pAttn.mmul(V);
            System.out.println(" MultiHeadAttention3 - attention - Arrays.toString(attnOutput.getShape()) 2- "+ Arrays.toString(attnOutput.getShape()));
            System.out.println(" MultiHeadAttention3 - attention - attnOutput.eval().shapeInfoToString() 2- "+ attnOutput.eval().shapeInfoToString());
            System.out.println(" MultiHeadAttention3 - attention - attnOutput.eval() 2- "+ attnOutput.eval());

            return attnOutput;
        }


//        private Pair<SDVariable, SDVariable> attention(SDVariable query, SDVariable key, SDVariable value, SDVariable mask, double dropout) {
////        private Pair<INDArray, INDArray> attention(INDArray query, INDArray key, INDArray value, INDArray mask, double dropout) {
//
//            SDVariable scores = sd.mmul(query, key.permute(0, 2, 1)).div(Math.sqrt(embSize));
////            SDVariable scores = sd.mmul(query, key.permute(0, 2, 1)).div(Math.sqrt(dModel));
////        INDArray scores = Nd4j.matmul(query, key.transpose(-2, -1)).div(Math.sqrt(dModel));
//            if (mask != null) {
//                INDArray scoresArray = scores.getArr();
//                applyMask(scoresArray, mask.getArr());
//                sd.associateArrayWithVariable(scoresArray, scores);
////            scores.setArray(scoresArray);
////            scores = scores.(mask.eq(0), -1e9);
////            scores = scores.maskedFill(mask.eq(0), -1e9);
//            }
//            SDVariable pAttn = sd.nn.softmax(scores, -1);
////        INDArray pAttn = Softmax.softmax(scores, -1);
//            if (dropout > 0.0) {
//                pAttn = dropout(pAttn, dropout);
////            pAttn = Dropout.dropout(pAttn, dropout);
//            }
//            SDVariable x = sd.mmul(pAttn, value);
//            return new Pair<>(x, pAttn);
//        }

        public static INDArray[] scaledDotProduct(INDArray q, INDArray k, INDArray v, INDArray mask) {
            long d_k = q.size(q.rank() - 1);
            INDArray attnLogits = q.mmul(k.permute(-2, -1)).div(Math.sqrt(d_k));
            if (mask != null) {
                applyMask(attnLogits, mask);
//            attnLogits = attnLogits.put(mask.eq(0), Nd4j.ones(mask.shape()).mul(-9e15));
//            attnLogits = attnLogits.putMaskedArray(mask.eq(0), Nd4j.ones(mask.shape()).mul(-9e15));
            }
            INDArray attention = Transforms.softmax(attnLogits);
//        INDArray attention = Transforms.softmax(attnLogits, 1);
            INDArray values = attention.mmul(v);
            INDArray[] scaledDotProductResult = null;
            scaledDotProductResult[0] = attention;
            scaledDotProductResult[1] = values;
            return scaledDotProductResult;
        }

        public static void applyMask(INDArray to, INDArray mask) {
            //Two possibilities exist: it's *per example* masking, or it's *per output* masking
            //These cases have different mask shapes. Per example: column vector. Per output: same shape as score array
            if (mask.isColumnVectorOrScalar()) {
                to.muliColumnVector(mask);
            } else if (Arrays.equals(to.shape(), mask.shape())) {
                to.muli(mask);
            } else {
                throw new IllegalStateException("Invalid mask array: per-example masking should be a column vector, "
                        + "per output masking arrays should be the same shape as the labels array. Mask shape: "
                        + Arrays.toString(mask.shape()) + ", output shape: " + Arrays.toString(to.shape()));
            }
        }


    }

    // Encoder Layer
    public static class EncoderLayer {
//        private double dropout;
//        private int embSize;
//        private int numHeads;
//        private int ffHiddenSize;

        SDVariable input = sd.placeHolder("input", DataType.DOUBLE, batch_size, -1, embSize);
//            SDVariable input = sd.placeHolder("input", Nd4j.create(DataType.DOUBLE, -1, -1, embSize));

        SDVariable dropoutVar = new SDVariable();
        SDVariable normAttn = new SDVariable();
        SDVariable attnResult = new SDVariable();
        SDVariable residualAttn = new SDVariable();
        SDVariable normFF = new SDVariable();
        SDVariable ffResult = new SDVariable();
        SDVariable residualFF = new SDVariable();

        MultiHeadAttention2 multiHeadAttention = new MultiHeadAttention2(numHeads, embSize, dropout);
        PointerwiseFeedforward pointerwiseFeedforward = new PointerwiseFeedforward(sd, embSize, ffHiddenSize, dropout);

//        LayerNorm layerNorm3 = new LayerNorm( embSize, 1e-5);
LayerNorm3 layerNorm3 = new LayerNorm3(sd, embSize, 1e-5);
        LayerNorm2 layerNorm2 = new LayerNorm2(sd, embSize, 1e-5);

//        private SameDiff sd;

        public EncoderLayer(int embSize1, int numHeads1, int ffHiddenSize1, double dropout1) {

//            class initializer
//
//            INPUT:
//            emb_size - (int) embedding size of the data
//            num_heads - (int) number of heads in multi head attention layer
//            ff_hidden_size - (int) size of the hidden layer for the feed forward network
//            dropout - (float) dropout percentage. Default value = 0.1

            embSize = embSize1;
            numHeads = numHeads1;
            ffHiddenSize = ffHiddenSize1;
            dropout = dropout1;
//            this.embSize = embSize1;
//            this.numHeads = numHeads1;
//            this.ffHiddenSize = ffHiddenSize1;
//            this.dropout = dropout1;

//            this.sd = SameDiff.create();
            buildModel();
        }

        private void buildModel() {

//                    # creating dropout layer
//            self.dropout = nn.Dropout(dropout)
//
//        # creating normalization layer for attention module
//            self.norm_attn = nn.LayerNorm(emb_size)
//        # creating normalization layer for feed forward layer
//            self.norm_ff = nn.LayerNorm(emb_size)
//
//        # creating object for multi head attention layer
//            self.attn = MultiHeadAttention(num_heads, emb_size, dropout)
//
//        # creating feed forward layer
//            self.ff = nn.Sequential(nn.Linear(emb_size, ff_hidden_size),
//                    nn.ReLU(),
//                    nn.Dropout(dropout),
//                    nn.Linear(ff_hidden_size, emb_size))

            mRandom = new Random();
            mRandomNumericalId = mRandom.nextInt(1000);

            HashMap<String, INDArray> placeholderData = new HashMap<>();
            placeholderData.put("input4", trainData.next().getFeatures());
            placeholderData.put("label4", trainData.next().getLabels());
            trainData.reset();

            // Input placeholder
//            SDVariable input = sd.placeHolder("input4", DataType.DOUBLE, batch_size, nIn, embSize);
            SDVariable input = sd.placeHolder("input4", DataType.DOUBLE, batch_size, nIn, -1);
//            SDVariable label = sd.placeHolder("label4", DataType.DOUBLE, batch_size, nOut, -1);
            SDVariable label = sd.placeHolder("label4", DataType.DOUBLE, batch_size, nOut, embSize);
//            SDVariable input = sd.placeHolder("input4", DataType.DOUBLE, batch_size, -1, embSize);
//            SDVariable label = sd.placeHolder("label4", DataType.DOUBLE, batch_size, -1, embSize);
//            SDVariable input = sd.placeHolder("input", Nd4j.create(DataType.DOUBLE, -1, -1, embSize));

            // creating dropout layer
            dropoutVar = sd.nn.dropout("dropout4"+" - "+mRandomNumericalId, input, dropout);
            System.out.println(" EncoderLayer - Arrays.toString(dropoutVar.getShape()) - "+ Arrays.toString(dropoutVar.getShape()));
            System.out.println(" EncoderLayer - dropoutVar.eval().shapeInfoToString() - "+ dropoutVar.eval(placeholderData).shapeInfoToString());

            // creating normalization layer for attention module
            layerNorm3 = new LayerNorm3(sd, embSize, 1e-5);
            normAttn = layerNorm3.forward(input);
//            SDVariable normAttn = sd.nn.layerNorm3("normAttn", input, true, 1, -1, -1);

            System.out.println(" EncoderLayer - Arrays.toString(normAttn.getShape()) - "+ Arrays.toString(normAttn.getShape()));
            System.out.println(" EncoderLayer - normAttn.eval().shapeInfoToString() - "+ normAttn.eval(placeholderData).shapeInfoToString());

//                    # creating normalization layer for feed forward layer
//            self.norm_ff = nn.LayerNorm(emb_size)
            layerNorm2 = new LayerNorm2(sd, embSize, 1e-5);

            // Multi-Head Attention Layer
            multiHeadAttention = new MultiHeadAttention2(numHeads, embSize, dropout);
            attnResult = multiHeadAttention.forward(sd, normAttn, normAttn, normAttn, null);
            System.out.println(" EncoderLayer - Arrays.toString(attnResult.getShape()) - "+ Arrays.toString(attnResult.getShape()));
            System.out.println(" EncoderLayer - attnResult.eval().shapeInfoToString() - "+ attnResult.eval(placeholderData).shapeInfoToString());

//                    # creating feed forward layer
//            self.ff = nn.Sequential(nn.Linear(emb_size, ff_hidden_size),
//                    nn.ReLU(),
//                    nn.Dropout(dropout),
//                    nn.Linear(ff_hidden_size, emb_size))

            INDArray dropoutVarArray = dropoutVar.getArr();
            INDArray dropoutVarArrayResized = Nd4j.create(attnResult.eval().shape()[0], attnResult.eval().shape()[1], attnResult.eval().shape()[2]);
            INDArray dropoutVarArrayResizedPopulated = dropoutVarArrayResized.assign(dropoutVarArray);
            SDVariable dropoutVarResizedPopulated = sd.var(dropoutVarArrayResizedPopulated);
//            SDVariable dropoutVarResizedPopulated = sd.var("dropoutVarResizedPopulated"+" - "+mRandomNumericalId, dropoutVarArrayResizedPopulated);
            // Residual Add for attention
            residualAttn = dropoutVarResizedPopulated.add(attnResult);
//            residualAttn = dropoutVar.add(attnResult);
            System.out.println(" EncoderLayer - Arrays.toString(residualAttn.getShape()) - "+ Arrays.toString(residualAttn.getShape()));
            System.out.println(" EncoderLayer - residualAttn.eval().shapeInfoToString() - "+ residualAttn.eval().shapeInfoToString());
//            System.out.println(" EncoderLayer - residualAttn.eval().shapeInfoToString() - "+ residualAttn.eval(placeholderData).shapeInfoToString());

            // creating normalization layer for feed forward layer
            layerNorm2 = new LayerNorm2(sd, embSize, 1e-5);
            normFF = layerNorm2.forward(residualAttn);
//            SDVariable normFF = sd.nn.layerNorm3("normFF", residualAttn, true, 1, -1, -1);
            System.out.println(" EncoderLayer - Arrays.toString(normFF.getShape()) - "+ Arrays.toString(normFF.getShape()));
            System.out.println(" EncoderLayer - normFF.eval().shapeInfoToString() - "+ normFF.eval().shapeInfoToString());

            // Feed Forward Network
            ffResult = pointerwiseFeedforward.forward(normFF);
//            SDVariable ffResult = feedForwardLayer(normFF);
            System.out.println(" EncoderLayer - Arrays.toString(ffResult.getShape()) - "+ Arrays.toString(ffResult.getShape()));
            System.out.println(" EncoderLayer - ffResult.eval().shapeInfoToString() - "+ ffResult.eval().shapeInfoToString());

            // Residual Add for feed forward layer
            residualFF = residualAttn.add(ffResult);
            System.out.println(" EncoderLayer - Arrays.toString(residualFF.getShape()) - "+ Arrays.toString(residualFF.getShape()));
            System.out.println(" EncoderLayer - residualFF.eval().shapeInfoToString() - "+ residualFF.eval().shapeInfoToString());
            System.out.println(" EncoderLayer - residualFF.eval() - "+ residualFF.eval());

//            sd.loss.logLoss("output", residualFF);
        }

//        private SDVariable feedForwardLayer(SDVariable input) {
//            // Perform feed forward computations here
//            // ...
//
//            // For this example, let's just use a simple feed forward layer with ReLU activation
//            SDVariable ffHidden = sd.nn.relu(sd.nn.linear("ff_hidden", input, sd.constant(Nd4j.eye(embSize))), 0);
//            return sd.nn.linear("output", ffHidden, sd.constant(Nd4j.eye(embSize)));
//        }

        public SDVariable forward(SDVariable input) {
//            public INDArray forward(INDArray input) {

//           forward pass through one encoder layer
//
//           INPUT:
//           x - (torch tensor) input data to the encoder layer. Shape = (B, N, C)
//
//           OUTPUT:
//           x - (torch tensor) output of the encoder layer. Shape = (B, N, C)

            mRandom = new Random();
            mRandomNumericalId = mRandom.nextInt(10000);

            HashMap<String, INDArray> placeholderData = new HashMap<>();
            placeholderData.put("input4", trainData.next().getFeatures());
            placeholderData.put("label4", trainData.next().getLabels());
            trainData.reset();

            System.out.println(" EncoderLayer - forward - Arrays.toString(input.getShape()) 0- "+ Arrays.toString(input.getShape()));
            System.out.println(" EncoderLayer - forward - input.eval().shapeInfoToString() 0- "+ input.eval(placeholderData).shapeInfoToString());
            System.out.println(" EncoderLayer - forward - input.eval(placeholderData) 0- "+ input.eval(placeholderData));
            System.out.println(" EncoderLayer - forward - input.eval() 0- "+ input.eval());

//        # sublayer 1: Input -> LayerNorm -> MultiHeadAttention -> Dropout -> ResidualAdd

            normAttn = layerNorm3.forward(input);
            System.out.println(" EncoderLayer - forward - Arrays.toString(normAttn.getShape()) 2- "+ Arrays.toString(normAttn.getShape()));
            System.out.println(" EncoderLayer - forward - normAttn.eval().shapeInfoToString() 2- "+ normAttn.eval(placeholderData).shapeInfoToString());

            SDVariable multiHeadAttentionOutput = multiHeadAttention.forward(sd, normAttn, normAttn, normAttn, null);
            System.out.println(" EncoderLayer - forward - Arrays.toString(multiHeadAttentionOutput.getShape()) - "+ Arrays.toString(multiHeadAttentionOutput.getShape()));
            System.out.println(" EncoderLayer - forward - multiHeadAttentionOutput.eval().shapeInfoToString() - "+ multiHeadAttentionOutput.eval().shapeInfoToString());

            SDVariable multiHeadAttentionOutputAfterDropout = dropout(multiHeadAttentionOutput, dropout);
            System.out.println(" EncoderLayer - forward - Arrays.toString(multiHeadAttentionOutputAfterDropout.getShape()) - "+ Arrays.toString(multiHeadAttentionOutputAfterDropout.getShape()));
            System.out.println(" EncoderLayer - forward - multiHeadAttentionOutputAfterDropout.eval().shapeInfoToString() - "+ multiHeadAttentionOutputAfterDropout.eval().shapeInfoToString());

            INDArray inputArray = input.getArr();
            INDArray inputArrayResized = Nd4j.create(multiHeadAttentionOutputAfterDropout.eval().shape()[0], multiHeadAttentionOutputAfterDropout.eval().shape()[1], multiHeadAttentionOutputAfterDropout.eval().shape()[2]);
            INDArray inputArrayResizedPopulated = inputArrayResized.assign(inputArray);
            SDVariable inputResizedPopulated = sd.var(inputArrayResizedPopulated);
//            SDVariable inputResizedPopulated = sd.var("inputResizedPopulated"+" - "+mRandomNumericalId, inputArrayResizedPopulated);

            SDVariable inputAddMultiheadAttentionOutputAfterDropout = inputResizedPopulated.add(multiHeadAttentionOutputAfterDropout);    // Shape = (B, N ,C)
//            SDVariable input = input.add(multiHeadAttentionOutputAfterDropout);    // Shape = (B, N ,C)
            System.out.println(" EncoderLayer - forward - Arrays.toString(inputAddMultiheadAttentionOutputAfterDropout.getShape()) 1- "+ Arrays.toString(inputAddMultiheadAttentionOutputAfterDropout.getShape()));
            System.out.println(" EncoderLayer - forward - inputAddMultiheadAttentionOutputAfterDropout.eval().shapeInfoToString() 1- "+ inputAddMultiheadAttentionOutputAfterDropout.eval().shapeInfoToString());

//SUBLAYER 1 PRODUCTION - END


//        # sublayer 2: Input -> LayerNorm -> FFN -> Dropout -> ResidualAdd

            SDVariable inputFinal0 = layerNorm2.forward(inputAddMultiheadAttentionOutputAfterDropout);
//            SDVariable inputFinal0 = layerNorm3.forward(inputAddMultiheadAttentionOutputAfterDropout);
            System.out.println(" EncoderLayer - forward - Arrays.toString(inputFinal0.getShape()) 1- "+ Arrays.toString(inputFinal0.getShape()));
            System.out.println(" EncoderLayer - forward - inputFinal0.eval().shapeInfoToString() 1- "+ inputFinal0.eval().shapeInfoToString());

            SDVariable inputFinal1 = sequential(inputFinal0, 0.1);
//            SDVariable inputFinal1 = pointerwiseFeedforward.forward(inputFinal0);

            System.out.println(" EncoderLayer - forward - Arrays.toString(inputFinal1.getShape()) 1- "+ Arrays.toString(inputFinal1.getShape()));
            System.out.println(" EncoderLayer - forward - inputFinal1.eval().shapeInfoToString() 1- "+ inputFinal1.eval().shapeInfoToString());


            SDVariable inputFinal2 = dropout(inputFinal1, dropout);
            System.out.println(" EncoderLayer - forward - Arrays.toString(inputFinal2.getShape()) 1- "+ Arrays.toString(inputFinal2.getShape()));
            System.out.println(" EncoderLayer - forward - inputFinal2.eval().shapeInfoToString() 1- "+ inputFinal2.eval().shapeInfoToString());

            SDVariable inputFinal = inputAddMultiheadAttentionOutputAfterDropout.add(inputFinal2);     // Shape = (B, N ,C)
//            input = input.add(dropout(pointerwiseFeedforward.forward(layerNorm3.forward(input)), dropout));     // Shape = (B, N ,C)
            System.out.println(" EncoderLayer - forward - Arrays.toString(inputFinal.getShape()) 1- "+ Arrays.toString(inputFinal.getShape()));
            System.out.println(" EncoderLayer - forward - inputFinal.eval().shapeInfoToString() 1- "+ inputFinal.eval().shapeInfoToString());
            System.out.println(" EncoderLayer - forward - inputFinal.eval() 1- "+ inputFinal.eval());

            return inputFinal;

//            sd.associateArrayWithVariable(input, sd.getVariable("input"));
//            sd.exec();
//            return sd.getArrForVarName("output").dup();

        }
    }


    // Encoder
    public static class Encoder {

//        private int embSize;
//        private int numHeads;
//        private int ffHiddenSize;
//        private int n;
//        private double dropout;

        public Encoder(int embSize1, int numHeads1, int ffHiddenSize1, int n1, double dropout1) {

//            class initializer
//
//        INPUT:
//            emb_size - (int) embedding size of the data
//            num_heads - (int) number of heads in multi head attention layer
//            ff_hidden_size - (int) size of the hidden layer for the feed forward network
//            n - (int) number of encoder layers
//            dropout - (float) dropout percentage. Default value = 0.1

            embSize = embSize1;
            numHeads = numHeads1;
            ffHiddenSize = ffHiddenSize1;
            n = n1;
            dropout = dropout1;
//            this.embSize = embSize1;
//            this.numHeads = numHeads1;
//            this.ffHiddenSize = ffHiddenSize1;
//            this.n = n1;
//            this.dropout = dropout1;

        }

        public SDVariable forward(SameDiff sd, SDVariable x) {

//            forward function to implement one pass through all layers of encoder
//
//            INPUT:
//            x - (torch tensor). input data. Shape = (B, N, C)
//
//            OUTPUT:
//            x - (torch tensor). output of the encoder block. Shape = (B, N, C)

//            HashMap<String, INDArray> placeholderData = new HashMap<>();
//            placeholderData.put("input4", trainData.next().getFeatures());
//            placeholderData.put("label4", trainData.next().getLabels());
            trainData.reset();

            System.out.println(" Encoder - forward - Arrays.toString(x.getShape()) - "+ Arrays.toString(x.getShape()));
            System.out.println(" Encoder - forward - x.eval(placeholderData).shapeInfoToString() - "+ x.eval().shapeInfoToString());
            System.out.println(" Encoder - forward - x.eval(placeholderData) 0- "+ x.eval());

            for (int i = 0; i < n; i++) {
                EncoderLayer layer = new EncoderLayer(embSize, numHeads, ffHiddenSize, dropout);
                x = layer.forward(x);
                System.out.println(" Encoder - forward - x.eval() - "+i+" == "+ x.eval());
            }

//            LayerNorm layerNorm3 = new LayerNorm( embSize, 1e-5);
            LayerNorm2 layerNorm2 = new LayerNorm2(sd, embSize, 1e-5);
//            LayerNorm layerNorm3 = new LayerNorm(sd, embSize, 1e-5);

            SDVariable norm = layerNorm2.forward(x);
//            SDVariable norm = layerNorm3.forward(x);
            System.out.println(" Encoder - forward - Arrays.toString(norm.getShape()) - "+ Arrays.toString(norm.getShape()));
            System.out.println(" Encoder - forward - norm.eval().shapeInfoToString() - "+ norm.eval().shapeInfoToString());
            System.out.println(" Encoder - forward - norm.eval() - "+ norm.eval());

//            SDVariable norm = sd.nn.layerNorm3(x, sd.var("encNormWeight", embSize));
            return norm;
        }
    }


    public static class DecoderLayer {

//        private double dropout;
//        private int embSize;
//        private int numHeads;
//        private int ffHiddenSize;

        //         # creating object for multi head self attention layer
//        self.attn = MultiHeadAttention(num_heads, emb_size, dropout)
//        # creating object for multi head encoder-decoder attention layer
//        self.enc_dec_attn = MultiHeadAttention(num_heads, emb_size, dropout)
//
//        # creating feed forward layer
//        self.ff = nn.Sequential(nn.Linear(emb_size, ff_hidden_size),
//                nn.ReLU(),
//                nn.Dropout(dropout),
//                nn.Linear(ff_hidden_size, emb_size))

        private MultiHeadAttention attn;
        private MultiHeadAttention3 encDecAttn;
        private PointerwiseFeedforward2 ff;

        //             # creating normalization layer for self attention module
//        self.norm_attn = nn.LayerNorm(emb_size)
//                # creating normalization layer for encoder-decoder attention module
//        self.norm_enc_dec = nn.LayerNorm(emb_size)
//                # creating normalization layer for feed forward layer
//        self.norm_ff = nn.LayerNorm(emb_size)

//        private LayerNorm normAttn = new LayerNorm( embSize, 1e-5);;
//        private LayerNorm normEncDec = new LayerNorm( embSize, 1e-5);
//        private LayerNorm normFF = new LayerNorm( embSize, 1e-5);
        private LayerNorm normAttn;;
        private LayerNorm normEncDec;
        private LayerNorm normFF;

        SDVariable dropoutLayer = new SDVariable();


//        # creating dropout layer
//        self.dropout = nn.Dropout(dropout)
//        SDVariable dropoutSDV = new SDVariable();

        public DecoderLayer(int embSize1, int numHeads1, int ffHiddenSize1, double dropout1) {
//            class initializer
//
//        INPUT:
//            emb_size - (int) embedding size of the data
//            num_heads - (int) number of heads in multi head attention layer
//            ff_hidden_size - (int) size of the hidden layer for the feed forward network
//            dropout - (float) dropout percentage. Default value = 0.1

            System.out.println(" - DecoderLayer - Printing sd information - 0");
            System.out.println(sd.summary());

            embSize = embSize1;
            numHeads = numHeads1;
            ffHiddenSize = ffHiddenSize1;
            dropout = dropout1;
//            this.embSize = embSize;
//            this.numHeads = numHeads;
//            this.ffHiddenSize = ffHiddenSize;
//            this.dropout = dropout;

//        # creating normalization layer for self attention module
//            self.norm_attn = nn.LayerNorm(emb_size)
//        # creating normalization layer for encoder-decoder attention module
//            self.norm_enc_dec = nn.LayerNorm(emb_size)
//        # creating normalization layer for feed forward layer
//            self.norm_ff = nn.LayerNorm(emb_size)

            normAttn = new LayerNorm(sd, embSize, 1e-5);;
            normEncDec = new LayerNorm(sd, embSize, 1e-5);
            normFF = new LayerNorm(sd, embSize, 1e-5);

//                    # creating object for multi head self attention layer
//            self.attn = MultiHeadAttention(num_heads, emb_size, dropout)
//        # creating object for multi head encoder-decoder attention layer
//            self.enc_dec_attn = MultiHeadAttention(num_heads, emb_size, dropout)

            attn = new MultiHeadAttention(numHeads, embSize, dropout);
            encDecAttn = new MultiHeadAttention3(numHeads, embSize, dropout);

//        # creating feed forward layer
//            self.ff = nn.Sequential(nn.Linear(emb_size, ff_hidden_size),
//                    nn.ReLU(),
//                    nn.Dropout(dropout),
//                    nn.Linear(ff_hidden_size, emb_size))

            ff = new PointerwiseFeedforward2(embSize, ffHiddenSize, dropout);

            System.out.println(" - DecoderLayer - Printing sd information - 1");
            System.out.println(sd.summary());

//            normAttn = new LayerNorm(sd, embSize, 1e-5);
//             normEncDec = new LayerNorm(sd, embSize, 1e-5);
//             normFF = new LayerNorm(sd, embSize, 1e-5);

//        # creating dropout layer
//            self.dropout = nn.Dropout(dropout)

            dropoutLayer = sd.var(Nd4j.scalar(dropout));

        }

        public SDVariable forward(SameDiff sd, SDVariable x, SDVariable encOutput, SDVariable sourceMask, SDVariable targetMask) {

            mRandom = new Random();
            mRandomNumericalId = mRandom.nextInt(10000);

            System.out.println(" DecoderLayer - x.eval() - "+ x.eval());
            System.out.println(" DecoderLayer - encOutput.eval() - "+ encOutput.eval());
            System.out.println(" DecoderLayer - sourceMask.eval() - "+ sourceMask.eval());
            System.out.println(" DecoderLayer - targetMask.eval() - "+ targetMask.eval());


//        # sublayer 1: Input -> LayerNorm -> MultiHeadAttention -> Dropout -> ResidualAdd
//            x = x + self.dropout(self.attn.forward(self.norm_attn(x),\
//                    self.norm_attn(x),self.norm_attn(x), target_mask))                          # Shape = (B, N ,C)

            SDVariable normAttnXOutput = normAttn.forward(x);
            System.out.println(" DecoderLayer - Arrays.toString(normAttnXOutput.getShape()) - "+ Arrays.toString(normAttnXOutput.getShape()));
            System.out.println(" DecoderLayer - normAttnXOutput.eval().shapeInfoToString() - "+ normAttnXOutput.eval().shapeInfoToString());

            SDVariable dropoutNormAttnXOutput = sd.nn.dropout(attn.forward(sd, normAttn.forward(x), normAttn.forward(x), normAttn.forward(x), targetMask), dropout);
            System.out.println(" DecoderLayer - Arrays.toString(dropoutNormAttnXOutput.getShape()) - "+ Arrays.toString(dropoutNormAttnXOutput.getShape()));
            System.out.println(" DecoderLayer - dropoutNormAttnXOutput.eval().shapeInfoToString() - "+ dropoutNormAttnXOutput.eval().shapeInfoToString());
            System.out.println(" DecoderLayer - dropoutNormAttnXOutput.eval() - "+ dropoutNormAttnXOutput.eval());

            INDArray Xarray = x.getArr();
            INDArray XarrayResized = Nd4j.create(dropoutNormAttnXOutput.eval().shape()[0], dropoutNormAttnXOutput.eval().shape()[1], dropoutNormAttnXOutput.eval().shape()[2]);
            INDArray XarrayResizedPopulated = XarrayResized.assign(Xarray);
            SDVariable xResizedPopulated = sd.var("xResizedPopulated"+" - "+mRandomNumericalId, XarrayResizedPopulated);

            SDVariable attnOutput = xResizedPopulated.add(dropoutNormAttnXOutput);
//SUBLAYER 1 - END

//        # sublayer 2: Input -> LayerNorm -> EncoderDecoderAttention -> Dropout -> ResidualAdd
//            x = x + self.dropout(self.enc_dec_attn.forward(self.norm_enc_dec(x),\
//                    self.norm_enc_dec(enc_output),self.norm_enc_dec(enc_output), source_mask))  # Shape = (B, N ,C)

//            SDVariable attnOutput = x.add(dropoutNormAttnXOutput);
//            SDVariable attnOutput = x.add(sd.nn.dropout(attn.forward(sd, normAttn.forward(x), normAttn.forward(x), normAttn.forward(x), targetMask), dropout));
            System.out.println(" DecoderLayer - Arrays.toString(attnOutput.getShape()) - "+ Arrays.toString(attnOutput.getShape()));
            System.out.println(" DecoderLayer - attnOutput.eval().shapeInfoToString() - "+ attnOutput.eval().shapeInfoToString());
            System.out.println(" DecoderLayer - attnOutput.eval() - "+ attnOutput.eval());

            SDVariable normEncDecAttnOutput = normEncDec.forward(attnOutput);
            System.out.println(" DecoderLayer - Arrays.toString(normEncDecAttnOutput.getShape()) - "+ Arrays.toString(normEncDecAttnOutput.getShape()));
            System.out.println(" DecoderLayer - normEncDecAttnOutput.eval().shapeInfoToString() - "+ normEncDecAttnOutput.eval().shapeInfoToString());
            System.out.println(" DecoderLayer - normEncDecAttnOutput.eval() - "+ normEncDecAttnOutput.eval());

            SDVariable normEncDecEncOutputOutput = normEncDec.forward(encOutput);
            System.out.println(" DecoderLayer - Arrays.toString(normEncDecEncOutputOutput.getShape()) - "+ Arrays.toString(normEncDecEncOutputOutput.getShape()));
            System.out.println(" DecoderLayer - normEncDecEncOutputOutput.eval().shapeInfoToString() - "+ normEncDecEncOutputOutput.eval().shapeInfoToString());
            System.out.println(" DecoderLayer - normEncDecEncOutputOutput.eval() - "+ normEncDecEncOutputOutput.eval());

            SDVariable encDecAttnOutput = encDecAttn.forward(sd, normEncDecAttnOutput, normEncDecEncOutputOutput, normEncDecEncOutputOutput, sourceMask);
            System.out.println(" DecoderLayer - Arrays.toString(encDecAttnOutput.getShape()) - "+ Arrays.toString(encDecAttnOutput.getShape()));
            System.out.println(" DecoderLayer - encDecAttnOutput.eval().shapeInfoToString() - "+ encDecAttnOutput.eval().shapeInfoToString());
            System.out.println(" DecoderLayer - encDecAttnOutput.eval() - "+ encDecAttnOutput.eval());

            SDVariable encDecAttnOutputDropout = sd.nn.dropout(encDecAttnOutput, dropout);
            System.out.println(" DecoderLayer - Arrays.toString(encDecAttnOutputDropout.getShape()) - "+ Arrays.toString(encDecAttnOutputDropout.getShape()));
            System.out.println(" DecoderLayer - encDecAttnOutputDropout.eval().shapeInfoToString() - "+ encDecAttnOutputDropout.eval().shapeInfoToString());
            System.out.println(" DecoderLayer - encDecAttnOutputDropout.eval() - "+ encDecAttnOutputDropout.eval());

            INDArray attnOutputarray = attnOutput.getArr();
            INDArray attnOutputarrayResized = Nd4j.create(encDecAttnOutputDropout.eval().shape()[0], encDecAttnOutputDropout.eval().shape()[1], encDecAttnOutputDropout.eval().shape()[2]);
            INDArray attnOutputarrayResizedPopulated = attnOutputarrayResized.assign(attnOutputarray);
            SDVariable attnOutputResizedPopulated = sd.var(attnOutputarrayResizedPopulated);
//            SDVariable attnOutputResizedPopulated = sd.var("attnOutputResizedPopulated"+" - "+mRandomNumericalId, attnOutputarrayResizedPopulated);

            SDVariable encDecAttnOutputFinal = attnOutputResizedPopulated.add(encDecAttnOutputDropout);
//            SDVariable encDecAttnOutputFinal = attnOutput.add(encDecAttnOutputDropout);
//            SDVariable encDecAttnOutput = attnOutput.add(sd.nn.dropout(encDecAttn.forward(sd, normEncDec.forward(attnOutput), normEncDec.forward(encOutput), normEncDec.forward(encOutput), sourceMask), dropout));
            System.out.println(" DecoderLayer - Arrays.toString(encDecAttnOutputFinal.getShape()) - "+ Arrays.toString(encDecAttnOutputFinal.getShape()));
            System.out.println(" DecoderLayer - encDecAttnOutputFinal.eval().shapeInfoToString() - "+ encDecAttnOutputFinal.eval().shapeInfoToString());
            System.out.println(" DecoderLayer - encDecAttnOutputFinal.eval() - "+ encDecAttnOutputFinal.eval());

//SUBLAYER 2 - END

//        # sublayer 3: Input -> LayerNorm -> FFN -> Dropout -> ResidualAdd
//            x = x + self.dropout(self.ff(self.norm_ff(x)))                                  # Shape = (B, N ,C)

            SDVariable encDecAttnOutputNormFf = normFF.forward(encDecAttnOutputFinal);
            System.out.println(" DecoderLayer - Arrays.toString(encDecAttnOutputNormFf.getShape()) - "+ Arrays.toString(encDecAttnOutputNormFf.getShape()));
            System.out.println(" DecoderLayer - encDecAttnOutputNormFf.eval().shapeInfoToString() - "+ encDecAttnOutputNormFf.eval().shapeInfoToString());
            System.out.println(" DecoderLayer - encDecAttnOutputNormFf.eval() - "+ encDecAttnOutputNormFf.eval());

            SDVariable encDecAttnOutputNormFfSequential = sequential(encDecAttnOutputNormFf, 0.1);
//            SDVariable encDecAttnOutputNorm = normFF.forward(encDecAttnOutputFinal);
            System.out.println(" DecoderLayer - Arrays.toString(encDecAttnOutputNormFfSequential.getShape()) - "+ Arrays.toString(encDecAttnOutputNormFfSequential.getShape()));
            System.out.println(" DecoderLayer - encDecAttnOutputNormFfSequential.eval().shapeInfoToString() - "+ encDecAttnOutputNormFfSequential.eval().shapeInfoToString());
            System.out.println(" DecoderLayer - encDecAttnOutputNormFfSequential.eval() - "+ encDecAttnOutputNormFfSequential.eval());

            SDVariable encDecAttnOutputNormFfDropout = sd.nn.dropout(encDecAttnOutputNormFfSequential, dropout);
            System.out.println(" DecoderLayer - Arrays.toString(encDecAttnOutputNormFfDropout.getShape()) - "+ Arrays.toString(encDecAttnOutputNormFfDropout.getShape()));
            System.out.println(" DecoderLayer - encDecAttnOutputNormFfDropout.eval().shapeInfoToString() - "+ encDecAttnOutputNormFfDropout.eval().shapeInfoToString());
            System.out.println(" DecoderLayer - encDecAttnOutputNormFfDropout.eval() - "+ encDecAttnOutputNormFfDropout.eval());

            SDVariable ffOutput = encDecAttnOutputFinal.add(encDecAttnOutputNormFfDropout);
//            SDVariable ffOutput = encDecAttnOutputFinal.add(sd.nn.dropout(ff.forward(normFF.forward(encDecAttnOutputFinal)), dropout));
            System.out.println(" DecoderLayer - Arrays.toString(ffOutput.getShape()) - "+ Arrays.toString(ffOutput.getShape()));
            System.out.println(" DecoderLayer - ffOutput.eval().shapeInfoToString() - "+ ffOutput.eval().shapeInfoToString());
            System.out.println(" DecoderLayer - ffOutput.eval() - "+ ffOutput.eval());

            return ffOutput;
        }
    }

//    Decoder:

    public static class Decoder {

        private List<DecoderLayer> decLayers;
        private LayerNorm norm;

        public Decoder(int embSize, int numHeads, int ffHiddenSize, int n, double dropout) {

//            class initializer
//
//        INPUT:
//            emb_size - (int) embedding size of the data
//            num_heads - (int) number of heads in multi head attention layer
//            ff_hidden_size - (int) size of the hidden layer for the feed forward network
//            n - (int) number of encoder layers
//            dropout - (float) dropout percentage. Default value = 0.1

//                    # defining LayerNorm for decoder end
//            norm = new LayerNorm( embSize, -1e6);
            norm = new LayerNorm(sd, embSize, -1e6);

            System.out.println(" - Decoder - Printing sd information - 0");
            System.out.println(sd.summary());

            //       # creating object for 1 decoder layer
            this.decLayers = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                this.decLayers.add(new DecoderLayer(embSize, numHeads, ffHiddenSize, dropout));
            }

            System.out.println(" - Decoder - Printing sd information - 1");
            System.out.println(sd.summary());

        }

        public SDVariable forward(SameDiff sd, SDVariable x, SDVariable encOutput, SDVariable sourceMask, SDVariable targetMask) {

//            x - (torch tensor) input data to the decoder block. Shape = (B, N, C)
//            enc_output - (torch tensor) output of the encoder block. Shape = (B, N, C)
//            source_mask - (torch tensor) mask for encoder-decoder attention layer
//            target_mask - (torch tensor) mask for decoder self attention layer
//
//            OUTPUT:
//            x - (torch tensor) output of the decoder layer. Shape = (B, N ,C)


            System.out.println(" Decoder - forward - x.eval() - "+ x.eval());
            System.out.println(" Decoder - forward - encOutput.eval() - "+ encOutput.eval());
            System.out.println(" Decoder - forward - sourceMask.eval() - "+ sourceMask.eval());
            System.out.println(" Decoder - forward - targetMask.eval() - "+ targetMask.eval());

            for (DecoderLayer layer : decLayers) {
                x = layer.forward(sd, x, encOutput, sourceMask, targetMask);
            }

            System.out.println(" Decoder - forward - norm.forward(x).eval() - "+ norm.forward(x).eval());

            return norm.forward(x);
        }
    }

//    public class PositionalEncoding {
//        private double dropout;
//        private INDArray pe;
//
//        public PositionalEncoding(int embSize, double dropout, int maxLen) {
//            this.dropout = dropout;
//
//            // Compute the positional encodings once in log space.
//            pe = Nd4j.zeros(maxLen, embSize);
//            INDArray position = Nd4j.arange(0, maxLen).reshape(maxLen, 1).castTo(Nd4j.defaultFloatingPointType());
//            INDArray divTerm = Nd4j.exp(Nd4j.arange(0, embSize, 2).mul(-(Math.log(10000.0) / embSize)));
//
//            pe.get(NDArrayIndex.all(), NDArrayIndex.interval(0, embSize, 2)).assign(Nd4j.sin(position.mul(divTerm)));
//            pe.get(NDArrayIndex.all(), NDArrayIndex.interval(1, embSize, 2)).assign(Nd4j.cos(position.mul(divTerm)));
//            pe = pe.reshape(1, maxLen, embSize);
//        }
//
//        public INDArray forward(INDArray x) {
//            // Assuming x has shape (B, N, C)
//            INDArray positionalEncoding = pe.get(NDArrayIndex.all(), NDArrayIndex.interval(0, x.size(1)));
//
//            // Add positional encodings to the input data
//            x.addi(positionalEncoding);
//            // Apply dropout
//            x.muli(Nd4j.rand(x.shape()).gt(dropout));
//
//            return x;
//        }
//
//
//    }
//
//    public class Embeddings {
//        // caching values
//        private int emb_size;
//        // creating liner layer for embedding input data
//        private SDVariable linear_embd;
//        // creating object for positional encoding
//        private org.deeplearning4j.examples.trajectorypredictiontransformer.PositionalEncoding pos_encoding;
//
//        public Embeddings(int input_size, int emb_size) {
//            // class initializer
//            // caching values
//            this.emb_size = emb_size;
//
//            // creating liner layer for embedding input data
//            this.linear_embd = SameDiff.create().var("linear_embd", Nd4j.randn(new int[]{input_size, emb_size}));
//
//            // creating object for positional encoding
//            this.pos_encoding = new org.deeplearning4j.examples.trajectorypredictiontransformer.PositionalEncoding(emb_size, 0.1, 5000);
//        }
//
//        public INDArray forward(INDArray x) {
//            // forward pass to generate input embeddings
//
//            // creating embeddings for input data
//            SDVariable inputVar = SameDiff.create().var("input_var", x);
//            SDVariable embVar = inputVar.mmul(linear_embd).mul(Math.sqrt(emb_size)); // Shape = (B, N, C)
//
//            // incorporating positional embeddings
//            SDVariable outputVar = pos_encoding.forward(embVar);
//
//            // Now we build a SameDiff instance to execute the forward computation
//            SameDiff sd = SameDiff.create();
//            sd.associateArrayWithVariable(x, inputVar);
//            sd.associateArrayWithVariable(Nd4j.create(new int[]{x.size(0), x.size(1), emb_size}), linear_embd);
//
//            INDArray result = sd.execAndEndResult();
//            return result;
//        }
//
//    }


    public static class OutputGenerator {

//        private int embSize;
        private int outputSize;
        private SDVariable outputGen = new SDVariable();

//    class to generate the output embeddings from the transformer's output

        public OutputGenerator(int embSize1, int outputSize1) {

//        class initializer
//
//        INPUT:
//        output_size - (int) size of the output data
//        emb_size - (int) size of the embedding

            embSize = embSize1;
            outputSize = outputSize1;
//            this.embSize = embSize;
//            this.outputSize = outputSize;

            mRandom = new Random();
            mRandomNumericalId = mRandom.nextInt(1000000);

//    creating linear layer for embedding input data

            SDVariable input = sd.var("outputGenInput"+" - "+mRandomNumericalId, Nd4j.create(embSize, outputSize));
//            outputGen = linear(input, weights, bias);

        }

        SDVariable forward(SDVariable x) {

            System.out.println(" OutputGenerator - forward - x.eval() 0- "+ x.eval());

//        forward pass to generate the output data
//
//        INPUT:
//        x - (torch tensor) input data from transformer. Shape = (B, N, output_dimension)
//
//        OUTPUT:
//        x - (torch tensor) output data. Shape = (B, N, output_size)

//            x =

            outputGen = linear(x, weights, bias);     //Shape =(B,N,output_size)

            System.out.println(" OutputGenerator - forward - Arrays.toString(outputGen.getShape()) - "+ Arrays.toString(outputGen.getShape()));
            System.out.println(" OutputGenerator - forward - outputGen.eval().shapeInfoToString() - "+ outputGen.eval().shapeInfoToString());
            System.out.println(" OutputGenerator - forward - outputGen.eval() 1- "+ outputGen.eval());

            return outputGen;
        }
    }

    public static class TFModel {

        private static Embeddings encoderEmbedding;
        private static Embeddings2 decoderEmbeddings;
        private static Encoder encoderBlock;
        private static Decoder decoderBlock;
        private static OutputGenerator outputGen;

        public TFModel(SameDiff sd1, int encoderIpSize1, int decoderIpSize1, int modelOpSize1, int embSize1,
                       int numHeads1, int ffHiddenSize1, int n1, double dropout1, SDVariable weights1, SDVariable bias1, int batch_size1,  int labelcount1) {

//            class initializer
//
//        INPUT:
//            encoder_ip_size - (int) dimension of the encoder input
//            decoder_ip_size - (int) dimension of the decoder input
//            model_op_size - (int) dimension of model's output
//            emb_size - (int) data embedding size for encoder and decoder
//            num_heads - (int) number of heads in multi head attention layer
//            ff_hidden_size - (int) size of the hidden layer for the feed forward network
//            n - (int) number of encoder layers
//            dropout - (float) dropout percentage. Default value = 0.1

            encoderIpSize = encoderIpSize1;
            decoderIpSize = decoderIpSize1;
            modelOpSize = modelOpSize1;
            embSize = embSize1;
            numHeads = numHeads1;
            ffHiddenSize = ffHiddenSize1;
            n = n1;
            dropout = dropout1;
            batch_size = batch_size1;
            labelCount = labelcount1;

            sd = sd1;
            weights = weights1;
            bias = bias1;

//            weights = weights;
//            bias = bias;

//            SDVariable input = sd.placeHolder("input", DataType.DOUBLE, -1, encoderIpSize, -1);
//            SDVariable label = sd.placeHolder("label", DataType.DOUBLE, -1, encoderIpSize, -1);

//            weights = sd.var("w1", new XavierInitScheme('c', encoderIpSize, modelOpSize), DataType.DOUBLE, modelOpSize, labelCount);
//            bias = sd.constant("b1", 0.05);

            // creating embeddings for encoder input
            encoderEmbedding = new Embeddings(sd, encoderIpSize, embSize);
            System.out.println(" TFModel -----------Instantiated encoderEmbeddings------------");
            // creating embeddings for decoder input
            decoderEmbeddings = new Embeddings2(sd, decoderIpSize, embSize);
            System.out.println(" TFModel -----------Instantiated decoderEmbeddings------------");

            // creating encoder block
            encoderBlock = new Encoder(embSize, numHeads, ffHiddenSize, n, dropout);
            System.out.println(" TFModel -----------Instantiated encoderBlock------------");
            // creating decoder block
            decoderBlock = new Decoder(embSize, numHeads, ffHiddenSize, n, dropout);
            System.out.println(" TFModel -----------Instantiated decoderBlock------------");

            // creating output generator
            outputGen = new OutputGenerator(embSize, modelOpSize);
            System.out.println(" TFModel -----------Instantiated outputGen------------");
        }

        public static SDVariable forward(SDVariable encInput, SDVariable decInput, SDVariable decSourceMask, SDVariable decTargetMask) {
//            public INDArray forward(INDArray encInput, INDArray decInput, INDArray decSourceMask, INDArray decTargetMask) {

//            forward pass for the transformer model
//
//            INPUT:
//            enc_input - (torch tensor) input data to the encoder block. Shape = (B, N, encoder_ip_size)
//            dec_input - (torch tensor) input data to the decoder block. Shape = (B, N, decoder_ip_size)
//            enc_output - (torch tensor) output of the encoder block. Shape = (B, N, emb_size)
//            source_mask - (torch tensor) mask for encoder-decoder attention layer
//            target_mask - (torch tensor) mask for decoder self attention layer
//
//            OUTPUT:
//            model_output - (torch tensor) output of the model. Shape = (B, N, model_op_size)


            System.out.println(" TFModel - forward - encInput.eval() - "+ encInput.eval());
            System.out.println(" TFModel - forward - decInput.eval() - "+ decInput.eval());
            System.out.println(" TFModel - forward - decSourceMask.eval() - "+ decSourceMask.eval());
            System.out.println(" TFModel - forward - decTargetMask.eval() - "+ decTargetMask.eval());

            System.out.println(" TFModel - forward - encInput.getShape().toString() -  "+ Arrays.toString(encInput.getShape()));

            System.out.println(" TFModel - forward - Arrays.toString(weights.getShape()) - "+ Arrays.toString(weights.getShape()));
            System.out.println(" TFModel - forward - weights.eval().shapeInfoToString() - "+ weights.eval().shapeInfoToString());

            SDVariable encEmbed = encoderEmbedding.forward(encInput);
//            System.out.println(" TFModel - forward - Arrays.toString(encEmbed.getShape()) - "+ Arrays.toString(encEmbed.getShape()));
//            System.out.println(" TFModel - forward - encEmbed.eval().shapeInfoToString() - "+ encEmbed.eval().shapeInfoToString());

            SDVariable encoderOutput = encoderBlock.forward(sd, encEmbed);

            System.out.println(" TFModel - forward - Arrays.toString(decInput.getShape()) 1- "+ Arrays.toString(decInput.getShape()));
            System.out.println(" TFModel - forward - decInput.eval().shapeInfoToString() 1- "+ decInput.eval().shapeInfoToString());
            System.out.println(" TFModel - forward - decInput.eval() 1- "+ decInput.eval());
            SDVariable decEmbed = decoderEmbeddings.forward(decInput);
            SDVariable decoderOutput = decoderBlock.forward(sd, decEmbed, encoderOutput, decSourceMask, decTargetMask);

            SDVariable modelOutput = outputGen.forward(decoderOutput);

            System.out.println(" TFModel - forward - modelOutput.eval() 1- "+ modelOutput.eval());
            return modelOutput;
        }
    }


    ////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////

    public static SDVariable subsequent_mask(int size)
    {

//    Function to compute the mask used in attention layer of decoder
//    INPUT:
//    size - (int) horizon size
//    OUTPUT:
//    mask - (torch tensor) boolean array to mask out the data in decoder

        mRandom = new Random();
        mRandomNumericalId = mRandom.nextInt(1000000);

    long[] attnShape = new long[]{1, size, size};
    SDVariable maskInput = sd.var("maskInput"+" - "+mRandomNumericalId, Nd4j.ones(attnShape));
    SDVariable mask = sd.linalg.triu(maskInput, 1);
    return mask;

    }


    public static void applyMask(INDArray to, INDArray mask) {
        //Two possibilities exist: it's *per example* masking, or it's *per output* masking
        //These cases have different mask shapes. Per example: column vector. Per output: same shape as score array
        if (mask.isColumnVectorOrScalar()) {
            to.muliColumnVector(mask);
        } else if (Arrays.equals(to.shape(), mask.shape())) {
            to.muli(mask);
        } else {
            throw new IllegalStateException("Invalid mask array: per-example masking should be a column vector, "
                    + "per output masking arrays should be the same shape as the labels array. Mask shape: "
                    + Arrays.toString(mask.shape()) + ", output shape: " + Arrays.toString(to.shape()));
        }
    }

    private INDArray createMask(INDArray decInput) {
        int batchSize = (int) decInput.size(0);
        int seqLength = (int) decInput.size(1);
        INDArray mask = Nd4j.ones(batchSize, seqLength, seqLength);

        for (int i = 0; i < seqLength; i++) {
            mask.get(NDArrayIndex.all(), NDArrayIndex.interval(i + 1, seqLength), NDArrayIndex.interval(0, i + 1)).assign(0.0);
        }

        return mask;
    }

    public static class LinearEmbedding {
        private final SDVariable weight;
//        private final INDArray weight;

        public LinearEmbedding(SameDiff sd, int inpSize, int dModel) {
            weight = sd.var("weight", Nd4j.randn(inpSize, dModel).muli(Math.sqrt(dModel)));
        }

        public SDVariable forward(SDVariable x) {
//            public INDArray forward(INDArray x) {
            return x.mmul(weight);
        }

//        public LinearEmbedding copy(Function<INDArray, INDArray> function) {
//            LinearEmbedding embedding = new LinearEmbedding(0, 0);
//            embedding.weight.assign(function.apply(weight));
//            return embedding;
//        }
//    }


    }

    }

//    public class TFModel {
//        private SameDiff sd;
//
//        public TFModel(int encoderIpSize, int decoderIpSize, int modelOpSize, int embSize,
//                       int numHeads, int ffHiddenSize, int n, double dropout) {
//            sd = SameDiff.create();
//
//            // creating embeddings for encoder input
//            SDVariable encInput = sd.placeHolder("encInput", VariableType.INPUT, encoderIpSize);
//            SDVariable encoderEmbedding = sd.nn.embed("encoderEmbedding", encInput, embSize);
//
//            // creating embeddings for decoder input
//            SDVariable decInput = sd.placeHolder("decInput", VariableType.INPUT, decoderIpSize);
//            SDVariable decoderEmbeddings = sd.nn.embed("decoderEmbeddings", decInput, embSize);
//
//            // creating encoder block
//            SDVariable encoderOutput = encoderBlock(encoderEmbedding, numHeads, ffHiddenSize, n, dropout);
//
//            // creating decoder block
//            SDVariable decSourceMask = sd.placeHolder("decSourceMask", VariableType.INPUT, encoderIpSize);
//            SDVariable decTargetMask = sd.placeHolder("decTargetMask", VariableType.INPUT, decoderIpSize);
//            SDVariable decoderOutput = decoderBlock(decoderEmbeddings, encoderOutput, decSourceMask, decTargetMask, numHeads, ffHiddenSize, n, dropout);
//
//            // creating output generator
//            SDVariable modelOutput = outputGenerator(decoderOutput, embSize, modelOpSize);
//            sd.loss.meanSquaredError("loss", modelOutput); // Set the loss function, you can change it to your desired loss function
//        }
//
//        private SDVariable encoderBlock(SDVariable encInput, int numHeads, int ffHiddenSize, int n, double dropout) {
//            SDVariable encoderOutput = encInput;
//            for (int i = 0; i < n; i++) {
//                encoderOutput = multiHeadAttention(encoderOutput, numHeads);
//                encoderOutput = feedForwardNetwork(encoderOutput, ffHiddenSize, dropout);
//            }
//            return encoderOutput;
//        }
//
//        private SDVariable decoderBlock(SDVariable decInput, SDVariable encoderOutput, SDVariable decSourceMask,
//                                        SDVariable decTargetMask, int numHeads, int ffHiddenSize, int n, double dropout) {
//            // Implement your decoder block logic here
//            // You will need to use the encoderOutput, decSourceMask, and decTargetMask to perform the self-attention
//            // and cross-attention operations
//            // Return the result as an SDVariable
//            return decInput;
//        }
//
//        public INDArray forward(INDArray encInput, INDArray decInput, INDArray decSourceMask, INDArray decTargetMask) {
//            sd.associateArrayWithVariable(encInput, sd.getVariable("encInput"));
//            sd.associateArrayWithVariable(decInput, sd.getVariable("decInput"));
//            sd.associateArrayWithVariable(decSourceMask, sd.getVariable("decSourceMask"));
//            sd.associateArrayWithVariable(decTargetMask, sd.getVariable("decTargetMask"));
//
//            sd.execAndEndResult();
//
//            // Get the output of the model
//            SDVariable modelOutput = sd.getVariable("outputGenerator");
//            INDArray outputArray = modelOutput.getArr();
//
//            return outputArray;
//        }
//
//
////        private SDVariable multiHeadAttention(SDVariable input, int numHeads) {
////            // Implement your multi-head attention logic here
////            // Return the result as an SDVariable
////            return input;
////        }
////
////        private SDVariable feedForwardNetwork(SDVariable input, int ffHiddenSize, double dropout) {
////            // Implement your feed-forward network logic here
////            // Return the result as an SDVariable
////            return input;
////        }
////        private SDVariable outputGenerator(SDVariable input, int embSize, int modelOpSize) {
////            // Implement your output generator logic here
////            // Return the result as an SDVariable
////            return input;
////        }
//
//
//    }


