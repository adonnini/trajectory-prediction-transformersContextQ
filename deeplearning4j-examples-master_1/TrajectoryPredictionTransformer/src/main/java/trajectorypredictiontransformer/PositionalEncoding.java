package trajectorypredictiontransformer;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;
import java.util.Random;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;

public class PositionalEncoding {

    int maxLen;
    int embSize;
    private double dropout;
//    private SDVariable pe = new SDVariable();
    private INDArray pe;
    private SameDiff sd = SameDiff.create();

    private Random mRandom;
    private int mRandomNumericalId;

    public PositionalEncoding(SameDiff sd, int embSize, double dropout, int maxLen) {

//    class initializer
//
//        INPUT:
//    emb_size - (int) size of the embedding
//    dropout - (float) dropout percentage. Default value = 0.1
//    max_len - (int) max positional length. Default value = 5000

        mRandom = new Random();
        mRandomNumericalId = mRandom.nextInt(1000);

        this.maxLen = maxLen;
        this.embSize = embSize;
        this.dropout = dropout;
        this.sd = sd;

        // Initialize pe with zeros
        pe = Nd4j.zeros(maxLen, embSize);

        System.out.println(" PositionalEncoding - maxLen - "+maxLen);
        System.out.println(" PositionalEncoding - embSize - "+embSize);

        maxLen = embSize;
        
        // Generate the position vector [0, 1, 2, ..., maxLen-1]
//        INDArray position = Nd4j.arange(0, maxLen).reshape(maxLen, 1).castTo(Nd4j.defaultFloatingPointType());
//
//        // Calculate div_term
//        INDArray divTerm = Nd4j.math.exp(
//                Nd4j.arange(0, embSize, 2).reshape(1, embSize/2).castTo(Nd4j.defaultFloatingPointType()).mul(-(Math.log(10000.0) / embSize))
//        );
//
//        // Calculate sin and cos values and assign them to pe
//        INDArray sinValues = Nd4j.math.sin(position.mmul(divTerm));
//        INDArray cosValues = Nd4j.math.cos(position.mmul(divTerm));
//
//        // Assign sin and cos values to pe
//        System.out.println("----- About to execute first pe.put -----");
//        pe.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(0, embSize, 2)}, sinValues);
//        System.out.println("----- About to execute second pe.put -----");
//        pe.put(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(1, embSize, 2)}, cosValues);


//        // Calculate the values for pe[:, 0::2]
//        INDArray range = Nd4j.arange(0, maxLen).reshape(maxLen, 1).castTo(Nd4j.defaultFloatingPointType());
//        INDArray div_term = Nd4j.arange(0, embSize, 2).reshape(1, embSize/2).castTo(Nd4j.defaultFloatingPointType());
//        double logTerm = Math.log(10000.0) / embSize;
//        INDArray pe1 = Nd4j.math().exp(range.mul(div_term.mul(-logTerm)));
//        pe1 = Nd4j.math.sin(pe1);
//
//        // Calculate the values for pe[:, 1::2]
//        INDArray pe2 = Nd4j.math().exp(range.mul(div_term.mul(-logTerm)));
//        pe2 = Nd4j.math.cos(pe2);
//
//        // Combine pe1 and pe2 into a single array
////        int numElements = (int) (pe1.size(1) + pe2.size(1));
////        INDArray pe = Nd4j.create(1, numElements);
//        pe.put(new INDArrayIndex[] { NDArrayIndex.all(), NDArrayIndex.interval(0, 2, embSize) }, pe1);
//        pe.put(new INDArrayIndex[] { NDArrayIndex.all(), NDArrayIndex.interval(1, 2, embSize) }, pe2);

        //        pe = torch.zeros(max_len, emb_size)
        INDArray peArray = Nd4j.zeros(maxLen, embSize);

//        position = torch.arange(0, max_len).unsqueeze(1).float()
        INDArray positionArray = Nd4j.arange(0, maxLen);
        SDVariable position = sd.var("positionForPe"+" - "+mRandomNumericalId, positionArray);
        sd.expandDims(position, 1);
        INDArray positionUnsqueezed = position.getArr();
        System.out.println(" PositionalEncoding - positionUnsqueezed.shape().length 1-  "+ positionUnsqueezed.shape().length);
        System.out.println(" PositionalEncoding - positionUnsqueezed.shape()[0] 1-  "+ positionUnsqueezed.shape()[0]);

//        div_term = torch.exp(torch.arange(0, emb_size, 2).float() * -(math.log(10000.0) / emb_size))
        INDArray div_termExpArgument1 = Nd4j.arange(0, embSize * 2, 2).castTo(Nd4j.defaultFloatingPointType());
//        INDArray div_termExpArgument1 = Nd4j.arange(0, embSize, 2).castTo(Nd4j.defaultFloatingPointType());
        double div_termExpArgument2 = Math.log(10000.0) / embSize;
        INDArray div_term = Nd4j.math().exp(div_termExpArgument1.mul( -div_termExpArgument2));
        System.out.println(" PositionalEncoding - div_term.shape().length 1-  "+ div_term.shape().length);
        System.out.println(" PositionalEncoding - div_term.shape()[0] 1-  "+ div_term.shape()[0]);

        INDArray pe1 = Nd4j.math.sin(positionUnsqueezed.mul(div_term));
        INDArray pe2 = Nd4j.math.cos(positionUnsqueezed.mul(div_term));

//        pe[:, 0::2] = torch.sin(position * div_term) //all in dim 0, in dim1 starting at position 0 step by 2
//        pe[:, 1::2] = torch.cos(position * div_term) //all in dim 0, in dim1 starting at position 1 step by 2
        peArray.put(new INDArrayIndex[] { NDArrayIndex.all(), NDArrayIndex.interval(0, 2, embSize) }, pe1);
        peArray.put(new INDArrayIndex[] { NDArrayIndex.all(), NDArrayIndex.interval(1, 2, embSize) }, pe2);


//        pe = pe.unsqueeze(0)
        SDVariable peUnsqueezed = sd.var("peUnsqueezed"+" - "+mRandomNumericalId, peArray);
        sd.expandDims(peUnsqueezed, 0);
        pe = peUnsqueezed.getArr();

        // Add a new dimension at the beginning
        pe = pe.reshape(1, maxLen, embSize);
        System.out.println(" PositionalEncoding - pe.shape().length 1-  "+ pe.shape().length);
        System.out.println(" PositionalEncoding - pe.shape()[0] 1-  "+ pe.shape()[0]);
        System.out.println(" PositionalEncoding - pe.shape()[1] 1-  "+ pe.shape()[1]);
        System.out.println(" PositionalEncoding - pe.shape()[2] 1-  "+ pe.shape()[2]);

//        // Compute the positional encodings once in log space.
//        pe = Nd4j.zeros(maxLen, embSize);
////        pe = sd.var("pe", Nd4j.zeros(maxLen, embSize));
//        INDArray position = Nd4j.arange(0, maxLen).reshape(maxLen, 1).castTo(Nd4j.defaultFloatingPointType());
//        INDArray divTerm = Nd4j.math.exp(Nd4j.arange(0, embSize, 2).mul(-(Math.log(10000.0) / embSize)));
//
//        pe.get(NDArrayIndex.all(), NDArrayIndex.interval(0, embSize, 2)).assign(Nd4j.math().sin(position.mul(divTerm)));
//        pe.get(NDArrayIndex.all(), NDArrayIndex.interval(1, embSize, 2)).assign(Nd4j.math.cos(position.mul(divTerm)));
//        pe = pe.reshape(1, maxLen, embSize);
    }

    public SDVariable forward(INDArray x) {
//        public INDArray forward(INDArray x) {

//        forward pass to generate positional embeddings
//
//        INPUT:
//        x - (torch tensor) embedded data. Shape = (B, N, C)
//
//        OUTPUT:
//        x - (torch tensor) positional embedded data. Shape = (B, N, C)

        mRandom = new Random();
        mRandomNumericalId = mRandom.nextInt(10000);

        System.out.println(" PositionalEncoding - x.shape().length 1-  "+ x.shape().length);
        System.out.println(" PositionalEncoding - x.shape()[0] 1-  "+ x.shape()[0]);
        System.out.println(" PositionalEncoding - x.shape()[1] 1-  "+ x.shape()[1]);
        System.out.println(" PositionalEncoding - x.shape()[2] 1-  "+ x.shape()[2]);

        // Assuming x has shape (B, N, C)
        INDArray positionalEncoding = pe.get(NDArrayIndex.all(), NDArrayIndex.interval(0, x.size(1)));
//        INDArray positionalEncoding = pe.get(NDArrayIndex.all(), NDArrayIndex.interval(0, x.size(2)));
        System.out.println(" PositionalEncoding - pe.shape().length 1-  "+ pe.shape().length);
        System.out.println(" PositionalEncoding - pe.shape()[0] 1-  "+ pe.shape()[0]);
        System.out.println(" PositionalEncoding - pe.shape()[1] 1-  "+ pe.shape()[1]);
        System.out.println(" PositionalEncoding - pe.shape()[2] 1-  "+ pe.shape()[2]);
        System.out.println(" PositionalEncoding - positionalEncoding.shape().length 1-  "+ positionalEncoding.shape().length);
        System.out.println(" PositionalEncoding - positionalEncoding.shape()[0] 1-  "+ positionalEncoding.shape()[0]);
        System.out.println(" PositionalEncoding - positionalEncoding.shape()[1] 1-  "+ positionalEncoding.shape()[1]);
        System.out.println(" PositionalEncoding - positionalEncoding.shape()[2] 1-  "+ positionalEncoding.shape()[2]);

        INDArray positionalEncodingReshaped = positionalEncoding.reshape(32, positionalEncoding.shape()[1], positionalEncoding.shape()[2] / 32);
//        INDArray positionalEncodingReshaped = positionalEncoding.reshape(32, 11, 16);
        System.out.println(" PositionalEncoding - positionalEncodingReshaped.shape().length 1-  "+ positionalEncodingReshaped.shape().length);
        System.out.println(" PositionalEncoding - positionalEncodingReshaped.shape()[0] 1-  "+ positionalEncodingReshaped.shape()[0]);
        System.out.println(" PositionalEncoding - positionalEncodingReshaped.shape()[1] 1-  "+ positionalEncodingReshaped.shape()[1]);
        System.out.println(" PositionalEncoding - positionalEncodingReshaped.shape()[2] 1-  "+ positionalEncodingReshaped.shape()[2]);

        INDArray positionalEncodingReshapedResized = Nd4j.create(x.shape()[0], x.shape()[1], x.shape()[2]);
        INDArray dropoutVarArrayResizedPopulated = positionalEncodingReshapedResized.assign(positionalEncodingReshaped);

        // Add positional encodings to the input data
        x.addi(dropoutVarArrayResizedPopulated);
//        x.addi(positionalEncodingReshaped);
        System.out.println(" PositionalEncoding - x.shape().length 1-  "+ x.shape().length);
        System.out.println(" PositionalEncoding - x.shape()[0] 2-  "+ x.shape()[0]);
        System.out.println(" PositionalEncoding - x.shape()[1] 2-  "+ x.shape()[1]);
        System.out.println(" PositionalEncoding - x.shape()[2] 2-  "+ x.shape()[2]);
        // Apply dropout
        INDArray peOutputArray = x.muli(Nd4j.rand(x.shape()).gt(dropout));
        System.out.println(" PositionalEncoding - peOutputArray.shape().length 1-  "+ peOutputArray.shape().length);
        System.out.println(" PositionalEncoding - peOutputArray.shape()[0] 3-  "+ peOutputArray.shape()[0]);
        System.out.println(" PositionalEncoding - peOutputArray.shape()[1] 3-  "+ peOutputArray.shape()[1]);
        System.out.println(" PositionalEncoding - peOutputArray.shape()[2] 3-  "+ peOutputArray.shape()[2]);

        SDVariable peOutput = sd.var("peOutput"+mRandomNumericalId, peOutputArray);
//        SDVariable peOutput = sd.var("peOutput"+" - "+mRandomNumericalId, peOutputArray);
        System.out.println(" PositionalEncoding - Arrays.toString(peOutput.getShape()) - "+ Arrays.toString(peOutput.getShape()));
        System.out.println(" PositionalEncoding - peOutput.eval().shapeInfoToString() - "+ peOutput.eval().shapeInfoToString());

        
//        SDVariable xSD = TransformerArchitectureModel.sd.var("xSDPositionalEncoding"+" - "+mRandomNumericalId, x);
//        System.out.println(" PositionalEncoding - Arrays.toString(xSD.getShape()) - "+ Arrays.toString(xSD.getShape()));
//        System.out.println(" PositionalEncoding - xSD.eval().shapeInfoToString() - "+ xSD.eval().shapeInfoToString());
//
//        SDVariable peAdd = TransformerArchitectureModel.sd.var("peAddPositionalEncoding"+" - "+mRandomNumericalId, pe.get(all(), NDArrayIndex.interval(0, x.size(1))));
//        System.out.println(" PositionalEncoding - Arrays.toString(peAdd.getShape()) - "+ Arrays.toString(peAdd.getShape()));
//        System.out.println(" PositionalEncoding - peAdd.eval().shapeInfoToString() - "+ peAdd.eval().shapeInfoToString());
//
//        SDVariable xSDreshaped = sd.reshape(xSD, peAdd.eval().shape()[0], peAdd.eval().shape()[1],peAdd.eval().shape()[2]);
//
//        SDVariable peOutput = TransformerArchitectureModel.sd.math.add(xSDreshaped, peAdd);
////        SDVariable peOutput = TransformerArchitectureModel.sd.math.add(xSD, peAdd);
//        System.out.println(" PositionalEncoding - Arrays.toString(peOutput.getShape()) - "+ Arrays.toString(peOutput.getShape()));
//        System.out.println(" PositionalEncoding - peOutput.eval().shapeInfoToString() - "+ peOutput.eval().shapeInfoToString());
//
//        x.addi(pe.get(all(), NDArrayIndex.interval(0, x.size(1))));

        return peOutput;

//        return x;
    }

//    public static void main(String[] args) {
//        int embSize = 512;
//        double dropout = 0.1;
//        int maxLen = 5000;
//
//        PositionalEncoding positionalEncoding = new PositionalEncoding(embSize, dropout, maxLen);
//
//        // Assuming x has shape (B, N, C)
//        INDArray x = Nd4j.rand(new int[]{2, 10, embSize}); // Example input with batch size 2 and sequence length 10
//
//        INDArray output = positionalEncoding.forward(x);
//        System.out.println(output);
//    }

}