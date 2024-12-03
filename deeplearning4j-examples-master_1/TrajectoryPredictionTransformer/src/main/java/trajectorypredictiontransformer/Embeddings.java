package trajectorypredictiontransformer;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Arrays;
import java.util.Random;

import static trajectorypredictiontransformer.TransformerArchitectureModel.*;

public class Embeddings {
    // caching values
    private int input_size;
    private int emb_size;
    // creating liner layer for embedding input data
    private SameDiff sd = SameDiff.create();
    private SDVariable linear_embd = new SDVariable();
    // creating object for positional encoding
    private PositionalEncoding pos_encoding;

    Random mRandom;
    int mRandomNumericalId;


    public Embeddings(SameDiff sd, int input_size, int emb_size) {
        // class initializer
//        INPUT:
//        input_size - (int) size of the input data
//        emb_size - (int) size of the embedding

        this.input_size = input_size;
        this.emb_size = emb_size;
        this.sd = sd;

        System.out.println(" Embeddings - input_size - "+input_size);
        System.out.println(" Embeddings - emb_size - "+emb_size);

        mRandom = new Random();
        mRandomNumericalId = mRandom.nextInt(10000);

        // creating liner layer for embedding input data
        this.linear_embd = TransformerArchitectureModel.sd.var("linear_embd"+" - "+mRandomNumericalId, Nd4j.randn(new int[] {input_size, emb_size}));

        // creating object for positional encoding
        this.pos_encoding = new PositionalEncoding(TransformerArchitectureModel.sd, emb_size, 0.1, emb_size / 2);

        System.out.println(" - Embeddings - Printing TransformerArchitectureModel.sd information");
        System.out.println(TransformerArchitectureModel.sd.summary());

    }

    public SDVariable forward(SDVariable x) {
//        public INDArray forward(INDArray x) {
        // forward pass to generate input embeddings
//
//        INPUT:
//        x - (torch tensor) input data. Shape = (B, N, input_dimension)
//
//        OUTPUT:
//        x - (torch tensor) embedded data. Shape = (B, N, C)

//        sd.withNameScope("Embeddings");

        System.out.println(" Embeddings - Arrays.toString(x.getShape()) - "+ Arrays.toString(x.getShape()));
        System.out.println(" Embeddings - x.eval().shapeInfoToString() - "+ x.eval().shapeInfoToString());
        System.out.println(" Embeddings - forward - x.eval() - "+ x.eval());
        System.out.println(" Embeddings - Arrays.toString(weights.getShape()) - "+ Arrays.toString(weights.getShape()));
        System.out.println(" Embeddings - weights.eval().shapeInfoToString() - "+ weights.eval().shapeInfoToString());
        System.out.println(" Embeddings - forward - weights.eval() - "+ weights.eval());

        // creating embeddings for input data
        x.setDataType(DataType.DOUBLE);
        SDVariable inputVar = linear(x, weights, bias).permute(0, 2, 1);
//        SDVariable inputVar = SameDiff.create().var("input_var", x);
        System.out.println(" Embeddings - Arrays.toString(inputVar.getShape()) - "+ Arrays.toString(inputVar.getShape()));
        System.out.println(" Embeddings - inputVar.eval().shapeInfoToString() - "+ inputVar.eval().shapeInfoToString());
//        System.out.println(" Embeddings - inputVar.eval().shape()[0] - "+ inputVar.eval().shape()[0]);
//        System.out.println(" Embeddings - inputVar.eval().shape()[1] - "+ inputVar.eval().shape()[1]);
//        System.out.println(" Embeddings - inputVar.eval().shape()[2] - "+ inputVar.eval().shape()[2]);
        System.out.println(" Embeddings - linear_embd.eval().shapeInfoToString() - "+ linear_embd.eval().shapeInfoToString());

        System.out.println(" Embeddings - batch_size - "+ batch_size);

        SDVariable linear_embdWithAddedDimension = sd.expandDims(linear_embd, 0);

        SDVariable linear_embdReshaped = linear_embd.reshape(batch_size, linear_embd.getShape()[0], linear_embd.getShape()[1]/batch_size);
//        SDVariable linear_embdReshaped = linear_embd.reshape(32, linear_embd.getShape()[0], linear_embd.getShape()[1]/32);

        System.out.println(" Embeddings - Arrays.toString(linear_embdReshaped.getShape()) - "+ Arrays.toString(linear_embdReshaped.getShape()));
        System.out.println(" Embeddings - linear_embdReshaped.eval().shapeInfoToString() - "+ linear_embdReshaped.eval().shapeInfoToString());

        System.out.println(" Embeddings - inputVar.eval().shapeInfoToString() - 1 - "+ inputVar.eval().shapeInfoToString());
        SDVariable inputVarPermuted = inputVar.permute(0, 2, 1);

        SDVariable inputVarReshaped = inputVar.reshape(batch_size, inputVar.eval().shape()[2]/3, inputVar.eval().shape()[1]*3);
//        SDVariable inputVarReshaped = inputVar.reshape(32, inputVar.eval().shape()[2]/3, inputVar.eval().shape()[1]*3);
        System.out.println(" Embeddings - inputVar.eval().shape()[1]*3 - "+ inputVar.eval().shape()[1]*3);
        System.out.println(" Embeddings - inputVar.eval().shape()[2]/3 - "+ inputVar.eval().shape()[2]/3);
        System.out.println(" Embeddings - Arrays.toString(inputVarReshaped.getShape()) - "+ Arrays.toString(inputVarReshaped.getShape()));
        System.out.println(" Embeddings - inputVarReshaped.eval().shapeInfoToString() - "+ inputVarReshaped.eval().shapeInfoToString());

        SDVariable embVar = inputVarReshaped.mmul(linear_embdReshaped).mul(Math.sqrt(emb_size)); // Shape = (B, N, C)
//        SDVariable embVar = inputVar.mmul(linear_embdReshaped).mul(Math.sqrt(emb_size)); // Shape = (B, N, C)
//        SDVariable embVar = inputVar.mmul(linear_embd).mul(Math.sqrt(emb_size)); // Shape = (B, N, C)
//        SDVariable embVar = inputVar.mmul(linear_embd).mul(Math.sqrt(emb_size)); // Shape = (B, N, C)
        System.out.println(" Embeddings - embVar.eval() - "+ embVar.eval());
        System.out.println(" Embeddings - embVar.eval().shapeInfoToString - "+ embVar.eval().shapeInfoToString());

        INDArray embVarArray = embVar.getArr();
        System.out.println(" Embeddings - embVarArray.shape().length 1-  "+ embVarArray.shape().length);
        System.out.println(" Embeddings - embVarArray.shape()[0] 1-  "+ embVarArray.shape()[0]);
        System.out.println(" Embeddings - embVarArray.shape()[1] 1-  "+ embVarArray.shape()[1]);
        System.out.println(" Embeddings - embVarArray.shape()[2] 1-  "+ embVarArray.shape()[2]);
        
        // incorporating positional embeddings
        SDVariable outputVar = TransformerArchitectureModel.sd.var(pos_encoding.forward(embVarArray));
        System.out.println(" Embeddings - Arrays.toString(outputVar.getShape()) - "+ Arrays.toString(outputVar.getShape()));
        System.out.println(" Embeddings - outputVar.eval().shapeInfoToString() - "+ outputVar.eval().shapeInfoToString());
        System.out.println(" Embeddings - outputVar.eval() - "+ outputVar.eval());

        return outputVar;

        // Now we build a SameDiff instance to execute the forward computation
//        SameDiff sd = SameDiff.create();

//        sd.associateArrayWithVariable(x, inputVar);
//        sd.associateArrayWithVariable(Nd4j.create(new int[] {x.size(0), x.size(1), emb_size}), linear_embd);
//
//        INDArray result = sd.execAndEndResult();
//        return result;

    }

//    public static void main(String[] args) {
//        int inputSize = 10; // Replace with your desired input size
//        int embSize = 8; // Replace with your desired embedding size
//
//        // Create an instance of the Embeddings class
//        Embeddings embeddings = new Embeddings(inputSize, embSize);
//
//        // Create a sample input tensor (B=2, N=5, input_dimension=10)
//        int B = 2;
//        int N = 5;
//        int input_dimension = 10;
//        INDArray input = Nd4j.randn(new int[] {B, N, input_dimension});
//
//        // Perform the forward pass to get embedded data
//        INDArray embeddedData = embeddings.forward(input);
//
//        // Print the embedded data
//        System.out.println("Embedded Data:");
//        System.out.println(embeddedData);
//    }

}

