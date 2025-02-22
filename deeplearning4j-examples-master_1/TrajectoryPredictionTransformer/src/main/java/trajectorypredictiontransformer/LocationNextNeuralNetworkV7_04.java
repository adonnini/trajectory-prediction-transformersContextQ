/* *****************************************************************************
 *
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package trajectorypredictiontransformer;

import au.com.bytecode.opencsv.CSVReader;
import au.com.bytecode.opencsv.CSVWriter;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.apache.commons.math3.stat.StatUtils;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.storage.FileStatsStorage;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.RefineryUtilities;
import org.nd4j.autodiff.listeners.impl.UIListener;
import org.nd4j.autodiff.listeners.records.History;
import org.nd4j.autodiff.samediff.NameScope;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.common.primitives.Pair;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.graph.ui.LogFileWriter;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMActivations;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMDataFormat;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMDirectionMode;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.config.LSTMLayerConfig;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.outputs.LSTMLayerOutputs;
import org.nd4j.linalg.api.ops.impl.layers.recurrent.weights.LSTMLayerWeights;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerSerializer;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.StandardizeSerializerStrategy;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.ops.NDBase;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.weightinit.impl.XavierInitScheme;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.swing.*;
import java.io.*;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.interval;

public class LocationNextNeuralNetworkV7_04 {
    private static final Logger Log = LoggerFactory.getLogger(LocationNextNeuralNetworkV7_04.class);

    //'baseDir': Base directory for the data. Change this if you want to save the data somewhere else
    private static File baseAssetDir = new File("src/main/assets/");
    private static File baseDir = new File("src/main/resources/uci/");
    private static File baseTrainDir = new File(baseDir, "train");
    private static File featuresDirTrain = new File(baseTrainDir, "features");
    private static File labelsDirTrain = new File(baseTrainDir, "labels");
    private static File baseTestDir = new File(baseDir, "test");
    private static File featuresDirTest = new File(baseTestDir, "features");
    private static File labelsDirTest = new File(baseTestDir, "labels");

    private static int contentAndLabels2Size = 0;
//    private static int contentAndLabelsSize = 0;

    private static int nItemsInDataSet = contentAndLabels2Size;   //75% train, 25% test
    //    private static int nItemsInDataSet = contentAndLabelsSize;   //75% train, 25% test
    private static int nTrain = (int)Math.round(nItemsInDataSet * .75);

    private static int nTest = nItemsInDataSet - nTrain;

    private static int lastTrainCount = 0;
    private static int lastTestCount = 0;

    private static SequenceRecordReader trainFeatures;
    private static SequenceRecordReader trainLabels;
    public static DataSetIterator trainData;
    private static SequenceRecordReader testFeatures;
    private static SequenceRecordReader testLabels;
    private static DataSetIterator testData;
    private static NormalizerStandardize normalizer;

    //Properties for dataset:
    public static int nIn = 6;
    public static int nOut = 2;
    public static int labelCount = 2;

    public static int batch_size;
//    public static int batch_size = lastTestCount/2;
//    public static int batch_size = 128;
//    public static int batch_size = 320;
//    public static int batch_size = 32;
//        public static int batch_size = 64;

    public static int encoder_ip_size = 6;
    //        public static int encoder_ip_size = 2;
    public static int decoder_ip_size = 3;
    public static int model_op_size = 6;
    //        public static int model_op_size = 3;
    public static int emb_size = 512;
    public static int num_heads = 8;
    public static int ff_hidden_size = 2048;
    public static int n = 1;
    //        public static int n = 6;
    public static double dropout=0.1;

    private static SameDiff sd = SameDiff.create();

    public static TransformerArchitectureModel.TFModel tf_model;

    public static SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, encoder_ip_size, -1);
    public static SDVariable label = sd.placeHolder("label", DataType.FLOAT, -1, nOut, -1);
//    public static SDVariable input = new SDVariable();
//    public static SDVariable label = new SDVariable();

    public static SDVariable weights = new SDVariable();
    public static SDVariable bias = new SDVariable();

    public static SDVariable out = new SDVariable();
    public static INDArray outArray;
    public static SDVariable outReduced = new SDVariable();
    public static String outReducedName = "";
    public static SDVariable labelInstance = new SDVariable();
    public static SDVariable labelResizedPopulated = new SDVariable();
    public static SDVariable lossMSE = new SDVariable();
    private static String outputVariable;
    private static Evaluation evaluation = new Evaluation();


    //    private static int miniBatchSize = 4;
//    private static int miniBatchSize = 16;
//    private static int miniBatchSize = 8;
//    private static int miniBatchSize = 256;
    private static int miniBatchSize = 128;
//    private static int miniBatchSize = 320;
//    private static int miniBatchSize = 32;

    private static int numLabelClasses = -1;
//    private static int numLabelClasses = 2;
//    private static int numLabelClasses = 1;

    private static long dim0 = 0L;
    private static long  dim1 = 0L;
    private static long dim2 = 0L;

    public static HashMap<String,INDArray> placeholderData = new HashMap<>();

    private static DataSet tNext;
    private static DataSet t;
    private static DataSet t2;

    private static INDArray w1Array;

    private static SDVariable mean = new SDVariable();
    private static SDVariable std = new SDVariable();

    private static SDVariable decInput = new SDVariable();
    private static SDVariable encInput = new SDVariable();
    private static SDVariable decSourceMask = new SDVariable();
    private static SDVariable decInputMask = new SDVariable();

    public static Random mRandom = new Random();
    public static int mRandomNumericalId = mRandom.nextInt(10000);

//    static UIServer uiServer = UIServer.getInstance();

    private static final String TAG = "LocationNextNeuralNetworkV7_04.java";

    public static void main(String[] args) throws Exception {

        downloadUCIData();

        System.out.println( TAG+" "+" - LocationNextNeuralNetworkV7_04.java - nTrain - "+nTrain);
        System.out.println( TAG+" "+" - LocationNextNeuralNetworkV7_04.java - nTest - "+nTest);
        System.out.println( TAG+" "+" - LocationNextNeuralNetworkV7_04.java - lastTrainCount - "+lastTrainCount);
        System.out.println( TAG+" "+" - LocationNextNeuralNetworkV7_04.java - lastTestCount - "+lastTestCount);

        Log.info(" - LocationNextNeuralNetworkV7_04.java - Log.info test --- ");

//        Nd4j.getExecutioner().enableVerboseMode(true);
//        Nd4j.getExecutioner().enableDebugMode(true);

        sameDiff3();

    }

    public static void sameDiff3() throws IOException, InterruptedException
    {

        prepareTrainingAndTestData();
//        System.out.println( TAG+" "+" sameDiff3 - lastTrainCount - 0 - "+lastTrainCount);
//        System.out.println( TAG+" "+" sameDiff3 - (lastTrainCount - lastTrainCount%miniBatchSize - 0) - 0 -"+(lastTrainCount - lastTrainCount%miniBatchSize - 1));
//        System.out.println( TAG+" "+" sameDiff3 - lastTestCount - 0 - "+lastTestCount);
//        System.out.println( TAG+" "+" sameDiff3 - (lastTestCount - lastTestCount%miniBatchSize - 0) - 0 -"+(lastTestCount - lastTestCount%miniBatchSize - 1));
//        // ----- Load the training data -----
//        trainFeatures = new CSVSequenceRecordReader();
////        trainFeatures.initialize(new NumberedFileInputSplit(featuresDirTrain.getAbsolutePath() + "/%d.csv", 0, lastTrainCount));
//        trainFeatures.initialize(new NumberedFileInputSplit(featuresDirTrain.getAbsolutePath() + "/%d.csv", 0, lastTrainCount - lastTrainCount%miniBatchSize - 1));
//        trainLabels = new CSVSequenceRecordReader();
////        trainLabels.initialize(new NumberedFileInputSplit(labelsDirTrain.getAbsolutePath() + "/%d.csv", 0, lastTrainCount));
//        trainLabels.initialize(new NumberedFileInputSplit(labelsDirTrain.getAbsolutePath() + "/%d.csv", 0, lastTrainCount - lastTrainCount%miniBatchSize - 1));
//
////        System.out.println( TAG+" "+" - trainFeatures.next().toString() - " + trainFeatures.next().toString());
////        System.out.println( TAG+" "+" - trainLabels.next().toString() - " + trainLabels.next().toString());
//
//        trainData = new SequenceRecordReaderDataSetIterator(trainFeatures, trainLabels, miniBatchSize, numLabelClasses,
//                true, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);
//
//
//        // ----- Load the test data -----
//        //Same process as for the training data.
//        testFeatures = new CSVSequenceRecordReader();
////        testFeatures.initialize(new NumberedFileInputSplit(featuresDirTest.getAbsolutePath() + "/%d.csv", 0, lastTestCount));
//        testFeatures.initialize(new NumberedFileInputSplit(featuresDirTest.getAbsolutePath() + "/%d.csv", 0, lastTestCount - lastTestCount%miniBatchSize - 1));
//        testLabels = new CSVSequenceRecordReader();
////        testLabels.initialize(new NumberedFileInputSplit(labelsDirTest.getAbsolutePath() + "/%d.csv", 0, lastTestCount));
//        testLabels.initialize(new NumberedFileInputSplit(labelsDirTest.getAbsolutePath() + "/%d.csv", 0, lastTestCount - lastTestCount%miniBatchSize - 1));
//
//        testData = new SequenceRecordReaderDataSetIterator(testFeatures, testLabels, miniBatchSize, numLabelClasses,
//                true, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);
//
//        trainData.reset();
//        testData.reset();
//        System.out.println( TAG+" "+" sameDiff3 - Printing traindata dataset shape - 0");
//        while(trainData.hasNext()) {
//            DataSet data = trainData.next();
//            System.out.println(Arrays.toString(data.getFeatures().shape()));
//        }
//        System.out.println( TAG+" "+" sameDiff3 - Printing testdata dataset shape - 0");
//        while(testData.hasNext()) {
//            DataSet data2 = testData.next();
//            System.out.println(Arrays.toString(data2.getFeatures().shape()));
//        }
//        trainData.reset();
//        testData.reset();
//
//        Log.info(" Printing traindata dataset shape - 0");
//        DataSet data = trainData.next();
//        System.out.println(Arrays.toString(data.getFeatures().shape()));
//
//        Log.info(" Printing testdata dataset shape - 0");
//        DataSet data2 = testData.next();
//        System.out.println(Arrays.toString(data2.getFeatures().shape()));
//
//        normalizer = new NormalizerStandardize();
//        normalizer.fitLabel(true);
//        normalizer.fit(trainData);           //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
//
//        trainData.reset();
//        testData.reset();
//
////        int index = 0;
//        while(trainData.hasNext()) {
////            ++index;
////            System.out.println( TAG+" "+" index - " + index);
//            normalizer.transform(trainData.next());     //Apply normalization to the training data
//        }
//
////        index = 0;
//        while(testData.hasNext()) {
////            ++index;
////            System.out.println( TAG+" "+" index - " + index);
//            normalizer.transform(testData.next());         //Apply normalization to the test data. This is using statistics calculated from the *training* set
//        }
//
//        trainData.reset();
//        testData.reset();
//
//        trainData.setPreProcessor(normalizer);
//        testData.setPreProcessor(normalizer);
//
//        System.out.println( TAG+" "+" sameDiff3 - Printing traindata dataset shape - 1");
//        while(trainData.hasNext()) {
//            data = trainData.next();
//            System.out.println(Arrays.toString(data.getFeatures().shape()));
//        }
//        System.out.println( TAG+" "+" sameDiff3 - Printing testdata dataset shape - 1");
//        while(testData.hasNext()) {
//            data2 = testData.next();
//            System.out.println(Arrays.toString(data2.getFeatures().shape()));
//        }
//
//        trainData.reset();
//        testData.reset();

        //ADD VISUALIZATION CODE HERE - START - <*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*>
        //Initialize the user interface backend
        UIServer uiServer = UIServer.getInstance();

        //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
        StatsStorage statsStorage = new InMemoryStatsStorage();         //Alternative: new FileStatsStorage(File), for saving and loading later

        String pathToStatsStorageFile = "src/main/assets/location_next_neural_network_v7_04_statsStorageLogFile.bin";
        File statsStorageLogFile = new File(pathToStatsStorageFile);
        StatsStorage fileStatsStorage = new FileStatsStorage(statsStorageLogFile);         //Alternative: new FileStatsStorage(File), for saving and loading later

        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
//        uiServer.attach(statsStorage);
        uiServer.attach(fileStatsStorage);

        //Then add the StatsListener to collect this information from the network, as it trains
//        int listenerFrequency = 1;
//        String pathToLogFile = "src/main/assets/location_next_neural_network_v7_04_logFile.bin";
//        File logFile = new File(pathToLogFile);
////        sd.setListeners(new ScoreListener(listenerFrequency, true, true));
//        UIListener mUIListener = UIListener.builder(logFile)
//                .plotLosses(1)
//                .trainEvaluationMetrics(outReduced.name(), 0, Evaluation.Metric.ACCURACY, Evaluation.Metric.F1)
////                .trainEvaluationMetrics("outReduced", 0, Evaluation.Metric.ACCURACY, Evaluation.Metric.F1)
//                .updateRatios(1)
//                .build();
//        sd.setListeners(mUIListener);

//        sd.setListeners(new ScoreListener(1));

//        try {
//            Ui(baseDir);
//        } catch (Exception e) {
//            throw new RuntimeException(e);
//        }

//        sd.setListeners( new StatsListener(statsStorage, listenerFrequency));
//        restored.setListeners(new StatsListener(statsStorage, listenerFrequency));
        //ADD VISUALIZATION CODE HERE - END - <*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*><*>

        t = trainData.next();
        dim0 = t.getFeatures().size(0);
        dim1 = t.getFeatures().size(1);
        dim2 = t.getFeatures().size(2);
        System.out.println( TAG+" "+" sameDiff3 - features - dim0 - 0 - "+dim0);
        System.out.println( TAG+" "+" sameDiff3 - features - dim1 - 0 - "+dim1);
        System.out.println( TAG+" "+" sameDiff3 - features - dim2 - 0 - "+dim2);
        trainData.reset();

        sd = SameDiff.create();

        placeholderData = new HashMap<>();
        trainData.reset();
        placeholderData.put("input",  trainData.next().getFeatures());
        placeholderData.put("label", trainData.next().getLabels());
        trainData.reset();

        createAndConfigureModel(placeholderData);
//        getConfiguration();
        trainData.reset();

        outputVariable = outReduced.name();
        outReducedName = outReduced.name();
//        String outputVariable = "out";
//            String outputVariable = "softmax";
        System.out.println( TAG+" "+" sameDiff3 - outReduced.name() 2-  "+ outReduced.name());

//        Evaluation evaluation = new Evaluation();
        RegressionEvaluation eval = new RegressionEvaluation(2);
//        RegressionEvaluation eval = new RegressionEvaluation();

//        int nEpochs = 1;
//        int nEpochs = 2;
        int nEpochs = 11;
        for (int i = 0; i < nEpochs; i++) {
            System.out.println( TAG+" "+" sameDiff3 - Epoch " + i + " starting. ");

            mRandom = new Random();
            mRandomNumericalId = mRandom.nextInt(100000000);

//            History history = sd.fit(trainData, 1);
////            List<Double> acc = history.trainingEval(Evaluation.Metric.ACCURACY);
////            System.out.println( TAG+" "+"Accuracy: " + acc);
//
//            trainData.reset();
//            System.out.println( TAG+" "+"Epoch " + i + " completed. ");

            System.out.println( TAG+" "+" sameDiff3 - Starting training --- ");

            testData.reset();
            trainData.reset();
            int testDataIndex = 0;
            int trainDataIndex = 0;
            HashMap<Integer,INDArray> placeholderDataLabels = new HashMap<>();
            HashMap<Integer,INDArray> placeholderDataInputs = new HashMap<>();
            HashMap<Integer,DataSet> placeholderDataDatasets = new HashMap<>();

//            System.out.println( TAG+" "+" sameDiff3 - testData.getLabels().size() - 0 - "+testData.getLabels().size());
            System.out.println( TAG+" "+" sameDiff3 - testData.totalOutcomes() - 0 - "+testData.totalOutcomes());
            System.out.println( TAG+" "+" sameDiff3 - trainData.totalOutcomes() - 0 - "+trainData.totalOutcomes());
            while(trainData.hasNext())
            {
                System.out.println( TAG+" "+" sameDiff3 - trainDataIndex - placeholderDataInputs - "+trainDataIndex);
                placeholderDataInputs.put(trainDataIndex,  trainData.next().getFeatures());
                trainDataIndex++;
//                ++trainDataIndex;
            }
            trainData.reset();
            trainDataIndex = 0;
            while(trainData.hasNext())
            {
                System.out.println( TAG+" "+" sameDiff3 - trainDataIndex - placeholderDataLabels - "+trainDataIndex);
                placeholderDataLabels.put(trainDataIndex, trainData.next().getLabels());
                trainDataIndex++;
//                ++trainDataIndex;
            }
            trainData.reset();
            trainDataIndex = 0;
            while(trainData.hasNext())
            {
                System.out.println( TAG+" "+" sameDiff3 - trainDataIndex - placeholderDataDatasets - "+trainDataIndex);
                placeholderDataDatasets.put(trainDataIndex, trainData.next());
                trainDataIndex++;
//                ++trainDataIndex;
            }
            trainData.reset();
            trainDataIndex = 0;
//            while(testData.hasNext())
//            {
//                System.out.println( TAG+" "+" sameDiff3 - testDataIndex - 0 - "+testDataIndex);
//                placeholderDataInputs.put(testDataIndex,  testData.next().getFeatures());
//                placeholderDataLabels.put(testDataIndex, testData.next().getLabels());
//                placeholderDataDatasets.put(testDataIndex, testData.next());
//                testDataIndex++;
////                ++testDataIndex;
//            }
            System.out.println( TAG+" "+" sameDiff3 - placeholderDataInputs.size() - 2 - "+placeholderDataInputs.size());
            System.out.println( TAG+" "+" sameDiff3 - placeholderDataLabels.size() - 2 - "+placeholderDataLabels.size());
            System.out.println( TAG+" "+" sameDiff3 - placeholderDataDatasets.size() - 2 - "+placeholderDataDatasets.size());

            Integer placeHolderDataKeyHolder = 0;
//            int loopCount = nEpochs;
            int loopCount = 0;

            ArrayList<DataSet> placeHolderDatasetArrayList = new ArrayList<DataSet>();
            int placeHolderDatasetArrayListRecordIndex = 0;
            for(Integer placeHolderDataDatasetsKey : placeholderDataDatasets.keySet())
            {
                placeHolderDatasetArrayList.add(placeholderDataDatasets.get(placeHolderDataDatasetsKey));
            }

//            int recordRemovalIndex = 0;
//            while(recordRemovalIndex < nEpochs && placeHolderDatasetArrayList.size() > 0)
//            {
//                placeHolderDatasetArrayList.remove(0);
//                ++recordRemovalIndex;
//                System.out.println( TAG+" "+" sameDiff3 - placeHolderDatasetArrayList.size() - 0 - "+placeHolderDatasetArrayList.size());
//            }

            int placeHolderDatasetArrayListSize = placeHolderDatasetArrayList.size();

                for(int j = 0; j < placeHolderDatasetArrayListSize; ++j)
//            for(int j = i; j < placeHolderDatasetArrayListSize; ++j)
//                for(int j = loopCount; j < placeHolderDatasetArrayListSize; ++j)
//            for(int j = placeHolderDatasetArrayListSize - 1; j >= 0; --j)
//                for(Integer placeHolderDataKey : placeholderDataInputs.keySet())
            {

                mRandom = new Random();
                mRandomNumericalId = mRandom.nextInt(100000000);

                NameScope ns = sd.withNameScope("Training"+" - "+mRandomNumericalId);

                INDArray featuresNext = null;
                INDArray labelsNext = null;
//                DataSet tNext = null;
//                DataSet t = null;

                ++loopCount;

                System.out.println( TAG+" "+" sameDiff3 - placeHolderDataKeyHolder - 1 - "+placeHolderDataKeyHolder);
                System.out.println( TAG+" "+" sameDiff3 - j - 1 - "+j);
//                System.out.println( TAG+" "+" sameDiff3 - placeHolderDataKey - 1 - "+placeHolderDataKey);


                if(((placeHolderDatasetArrayListSize - 1) == 0))
                {
                    System.out.println( TAG+" "+" sameDiff3 -----------((placeHolderDatasetArrayListSize - 1) == 0)------------");

                    tNext = placeHolderDatasetArrayList.get(j);
                    System.out.println( TAG+" "+" sameDiff3 - tNext.getFeatures().size(0) - 2a - "+tNext.getFeatures().size(0));
                    System.out.println( TAG+" "+" sameDiff3 - tNext.getFeatures().size(1) - 2a - "+tNext.getFeatures().size(1));
                    System.out.println( TAG+" "+" sameDiff3 - tNext.getFeatures().size(2) - 2a - "+tNext.getFeatures().size(2));

                    t = placeholderDataDatasets.get(j);
//                INDArray features = placeholderDataInputs.get(placeHolderDataKey);
//                INDArray labels = placeholderDataLabels.get(placeHolderDataKey);
//                DataSet t = placeholderDataDatasets.get(placeHolderDataKey);

//            dim0 = t2.getFeatures().size(0);
//            dim1 = t2.getFeatures().size(1);
//            dim2 = t2.getFeatures().size(2);
//            System.out.println( TAG+" "+" sameDiff3 - features - dim0 - t2 - "+dim0);
//            System.out.println( TAG+" "+" sameDiff3 - features - dim1 - t2 - "+dim1);
//            System.out.println( TAG+" "+" sameDiff3 - features - dim2 - t2 - "+dim2);

                    dim0 = t.getFeatures().size(0);
                    dim1 = t.getFeatures().size(1);
                    dim2 = t.getFeatures().size(2);
                    System.out.println( TAG+" "+" sameDiff3 - features - dim0 - 1a - "+dim0);
                    System.out.println( TAG+" "+" sameDiff3 - features - dim1 - 1a - "+dim1);
                    System.out.println( TAG+" "+" sameDiff3 - features - dim2 - 1a - "+dim2);

                }
                if(((placeHolderDatasetArrayListSize - 1) > 0) && (j == 0))
                {
                    System.out.println( TAG+" "+" sameDiff3 -----------((placeHolderDatasetArrayListSize - 1) > 0) && (j == 0)------------");

//                    tNext = placeHolderDatasetArrayList.get(j);
                    tNext = placeHolderDatasetArrayList.get(j+1);
                    System.out.println( TAG+" "+" sameDiff3 - tNext.getFeatures().size(0) - 2b - "+tNext.getFeatures().size(0));
                    System.out.println( TAG+" "+" sameDiff3 - tNext.getFeatures().size(1) - 2b - "+tNext.getFeatures().size(1));
                    System.out.println( TAG+" "+" sameDiff3 - tNext.getFeatures().size(2) - 2b - "+tNext.getFeatures().size(2));

                    t = placeholderDataDatasets.get(j);
//                INDArray features = placeholderDataInputs.get(placeHolderDataKey);
//                INDArray labels = placeholderDataLabels.get(placeHolderDataKey);
//                DataSet t = placeholderDataDatasets.get(placeHolderDataKey);

//            dim0 = t2.getFeatures().size(0);
//            dim1 = t2.getFeatures().size(1);
//            dim2 = t2.getFeatures().size(2);
//            System.out.println( TAG+" "+" sameDiff3 - features - dim0 - t2 - "+dim0);
//            System.out.println( TAG+" "+" sameDiff3 - features - dim1 - t2 - "+dim1);
//            System.out.println( TAG+" "+" sameDiff3 - features - dim2 - t2 - "+dim2);

                    dim0 = t.getFeatures().size(0);
                    dim1 = t.getFeatures().size(1);
                    dim2 = t.getFeatures().size(2);
                    System.out.println( TAG+" "+" sameDiff3 - features - dim0 - 1b - "+dim0);
                    System.out.println( TAG+" "+" sameDiff3 - features - dim1 - 1b - "+dim1);
                    System.out.println( TAG+" "+" sameDiff3 - features - dim2 - 1b - "+dim2);

                }
                else if(((placeHolderDatasetArrayListSize - 1) > 0) && (j == (placeHolderDatasetArrayListSize - 1)))
                {
                    System.out.println( TAG+" "+" sameDiff3 -----------((placeHolderDatasetArrayListSize - 1) > 0) && (j == (placeHolderDatasetArrayListSize - 1))------------");

                    tNext = placeHolderDatasetArrayList.get(j);
                    System.out.println( TAG+" "+" sameDiff3 - tNext.getFeatures().size(0) - 2c - "+tNext.getFeatures().size(0));
                    System.out.println( TAG+" "+" sameDiff3 - tNext.getFeatures().size(1) - 2c - "+tNext.getFeatures().size(1));
                    System.out.println( TAG+" "+" sameDiff3 - tNext.getFeatures().size(2) - 2c - "+tNext.getFeatures().size(2));

                }
                else if(((placeHolderDatasetArrayListSize - 1) > 0) && (j < (placeHolderDatasetArrayListSize - 1)))
                {
                    System.out.println( TAG+" "+" sameDiff3 -----------((placeHolderDatasetArrayListSize - 1) > 0) && (j < (placeHolderDatasetArrayListSize - 1))------------");

                    tNext = placeHolderDatasetArrayList.get(j);
//                    tNext = placeHolderDatasetArrayList.get(j+1);
                    System.out.println( TAG+" "+" sameDiff3 - tNext.getFeatures().size(0) - 2d - "+tNext.getFeatures().size(0));
                    System.out.println( TAG+" "+" sameDiff3 - tNext.getFeatures().size(1) - 2d - "+tNext.getFeatures().size(1));
                    System.out.println( TAG+" "+" sameDiff3 - tNext.getFeatures().size(2) - 2d - "+tNext.getFeatures().size(2));
                    System.out.println( TAG+" "+" sameDiff3 - tNext.getLabels().size(0) - 2d - "+tNext.getLabels().size(0));
                    System.out.println( TAG+" "+" sameDiff3 - tNext.getLabels().size(1) - 2d - "+tNext.getLabels().size(1));
                    System.out.println( TAG+" "+" sameDiff3 - tNext.getLabels().size(2) - 2d - "+tNext.getLabels().size(2));

                    t = placeholderDataDatasets.get(j);

                    dim0 = t.getFeatures().size(0);
                    dim1 = t.getFeatures().size(1);
                    dim2 = t.getFeatures().size(2);
                    System.out.println( TAG+" "+" sameDiff3 - features - dim0 - 2d - "+dim0);
                    System.out.println( TAG+" "+" sameDiff3 - features - dim1 - 2d - "+dim1);
                    System.out.println( TAG+" "+" sameDiff3 - features - dim2 - 2d - "+dim2);

                    dim0 = t.getLabels().size(0);
                    dim1 = t.getLabels().size(1);
                    dim2 = t.getLabels().size(2);
                    System.out.println( TAG+" "+" sameDiff3 - labels - dim0 - 2d - "+dim0);
                    System.out.println( TAG+" "+" sameDiff3 - labels - dim1 - 2d - "+dim1);
                    System.out.println( TAG+" "+" sameDiff3 - labels - dim2 - 2d - "+dim2);


//                    INDArray outReducedArray = outArray.get(NDArrayIndex.interval(0, tNext.getLabels().size(0)), NDArrayIndex.interval(0, tNext.getLabels().size(1)), NDArrayIndex.interval(0, tNext.getLabels().size(2)));
////                    INDArray outReducedArray = outArray.get(NDArrayIndex.interval(0, tNext.getFeatures().size(0)), NDArrayIndex.interval(0, tNext.getFeatures().size(1)), NDArrayIndex.interval(0, tNext.getFeatures().size(2)));
//                    System.out.println( TAG+" "+" sameDiff3 - outReducedArray.shape().length 3-  "+ outReducedArray.shape().length);
//                    System.out.println( TAG+" "+" sameDiff3 - outReducedArray.shape()[0] 3-  "+ outReducedArray.shape()[0]);
//                    System.out.println( TAG+" "+" sameDiff3 - outReducedArray.shape()[1] 3-  "+ outReducedArray.shape()[1]);
//                    System.out.println( TAG+" "+" sameDiff3 - outReducedArray.shape()[2] 3-  "+ outReducedArray.shape()[2]);
//                    System.out.println( TAG+" "+" sameDiff3 - outReducedArray 3-  "+ outReducedArray);
//
//                    sd.associateArrayWithVariable(outReducedArray, outReduced);
////                    outReduced = sd.var("outReduced"+" - "+mRandomNumericalId, outReducedArray);
//                    System.out.println( TAG+" "+" sameDiff3 - outReduced.getShape()[0] 3-  "+ outReduced.getShape()[0]);
//                    System.out.println( TAG+" "+" sameDiff3 - outReduced.getShape()[1] 3-  "+ outReduced.getShape()[1]);
//                    System.out.println( TAG+" "+" sameDiff3 - outReduced.getShape()[2] 3-  "+ outReduced.getShape()[2]);
//                    System.out.println( TAG+" "+" sameDiff3 - outReduced.getArr() 3-  "+ outReduced.getArr());
//                    System.out.println( TAG+" "+" sameDiff3 - outReduced.name() 3-  "+ outReduced.name());

                }

//sameDiff3 DATASET tNext  - START - _+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+
//                if(j < placeHolderDatasetArrayListSize - 1)
////                if(placeHolderDataKeyHolder.equals(j))
////                    if(placeHolderDataKeyHolder.equals(placeHolderDataKey))
//                {
//                    featuresNext = placeholderDataInputs.get(j + 1);
//                    labelsNext = placeholderDataLabels.get(j + 1);
//                    tNext = placeHolderDatasetArrayList.get(j + 1);
////                    featuresNext = placeholderDataInputs.get(j - 1);
////                    labelsNext = placeholderDataLabels.get(j - 1);
////                    tNext = placeHolderDatasetArrayList.get(j - 1);
////                    tNext = placeholderDataDatasets.get(j + 1);
////                    featuresNext = placeholderDataInputs.get(placeHolderDataKey + 1);
////                    labelsNext = placeholderDataLabels.get(placeHolderDataKey + 1);
////                    tNext = placeholderDataDatasets.get(placeHolderDataKey + 1);
//                    ++placeHolderDataKeyHolder;
//                    System.out.println( TAG+" "+" sameDiff3 - tNext.getFeatures().size(0) - 1 - "+tNext.getFeatures().size(0));
//                    System.out.println( TAG+" "+" sameDiff3 - tNext.getFeatures().size(1) - 1 - "+tNext.getFeatures().size(1));
//                    System.out.println( TAG+" "+" sameDiff3 - tNext.getFeatures().size(2) - 1 - "+tNext.getFeatures().size(2));
//
//                }
//sameDiff3 DATASET tNext  - END - _+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+

//sameDiff3 DATASET t  - START - =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
//                INDArray features = placeholderDataInputs.get(j);
//                INDArray labels = placeholderDataLabels.get(j);
//                DataSet t = placeholderDataDatasets.get(j);
////                INDArray features = placeholderDataInputs.get(placeHolderDataKey);
////                INDArray labels = placeholderDataLabels.get(placeHolderDataKey);
////                DataSet t = placeholderDataDatasets.get(placeHolderDataKey);
//
//                dim0 = t.getFeatures().size(0);
//                dim1 = t.getFeatures().size(1);
//                dim2 = t.getFeatures().size(2);
//                System.out.println( TAG+" "+" sameDiff3 - features - dim0 - 1 - "+dim0);
//                System.out.println( TAG+" "+" sameDiff3 - features - dim1 - 1 - "+dim1);
//                System.out.println( TAG+" "+" sameDiff3 - features - dim2 - 1 - "+dim2);
//
////                placeholderData = new HashMap<>();
////                placeholderData.put("input",  t.getFeatures());
////                placeholderData.put("label", t.getLabels());
////                System.out.println( TAG+" "+" sameDiff3 - ======================================================= - ");
////                System.out.println( TAG+" "+" sameDiff3 - Arrays.toString(label.eval().shape()) - "+ Arrays.toString(label.eval(placeholderData).shape()));
//
////                createAndConfigureModel();
//
////                outArray = tf_model.forward(encInput, decInput, decSourceMask, decInputMask).getArr();
////                System.out.println( TAG+" "+" sameDiff3 - outArray.shape().length 2-  "+ outArray.shape().length);
////                System.out.println( TAG+" "+" sameDiff3 - outArray.shape()[0] 2-  "+ outArray.shape()[0]);
////                System.out.println( TAG+" "+" sameDiff3 - outArray.shape()[1] 2-  "+ outArray.shape()[1]);
////                System.out.println( TAG+" "+" sameDiff3 - outArray.shape()[2] 2-  "+ outArray.shape()[2]);
//
//                dim0 = t.getLabels().size(0);
//                dim1 = t.getLabels().size(1);
//                dim2 = t.getLabels().size(2);
//                System.out.println( TAG+" "+" sameDiff3 - labels - dim0 - 2 - "+dim0);
//                System.out.println( TAG+" "+" sameDiff3 - labels - dim1 - 2 - "+dim1);
//                System.out.println( TAG+" "+" sameDiff3 - labels - dim2 - 2 - "+dim2);
//
////                INDArray outReducedArray = outArray.get(NDArrayIndex.all(), NDArrayIndex.interval(0, dim1), NDArrayIndex.interval(0, dim2));
//////                INDArray outReducedArray = outArray.get(NDArrayIndex.all(), NDArrayIndex.interval(0, labels.shape()[1]), NDArrayIndex.interval(0, labels.shape()[2]));
////                System.out.println( TAG+" "+" sameDiff3 - outReducedArray.shape().length 1-  "+ outReducedArray.shape().length);
////                System.out.println( TAG+" "+" sameDiff3 - outReducedArray.shape()[0] 2-  "+ outReducedArray.shape()[0]);
////                System.out.println( TAG+" "+" sameDiff3 - outReducedArray.shape()[1] 2-  "+ outReducedArray.shape()[1]);
////                System.out.println( TAG+" "+" sameDiff3 - outReducedArray.shape()[2] 2-  "+ outReducedArray.shape()[2]);
//
////                outReduced = sd.var(outReducedArray);
//////                outReduced = sd.var("outReduced2"+" - "+mRandomNumericalId, outReducedArray);
//////                outReduced = sd.var("outReduced"+" - "+mRandomNumericalId, outReducedArray);
////                System.out.println( TAG+" "+" sameDiff3 - Arrays.toString(outReduced.getShape()) 4- "+ Arrays.toString(outReduced.getShape()));
////                System.out.println( TAG+" "+" sameDiff3 - outReduced.eval().shapeInfoToString() 4- "+ outReduced.eval().shapeInfoToString());
//
////                labelInstance = sd.var("labelInstance"+" - "+mRandomNumericalId, t.getLabels());
////                labelInstance = sd.var(t.getLabels());
////                labelInstance = sd.var("", labels);
////                System.out.println( TAG+" "+" sameDiff3 - Arrays.toString(labelInstance.getShape()) 4- "+ Arrays.toString(labelInstance.getShape()));
////                System.out.println( TAG+" "+" sameDiff3 - labelInstance.eval().shapeInfoToString() 4- "+ labelInstance.eval().shapeInfoToString());
//sameDiff3 DATASET t  - END - =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

                System.out.println( TAG+" "+" sameDiff3 - outReduced.name() 4-  "+ outReduced.name());
                System.out.println( TAG+" "+" sameDiff3 - outputVariable 0- "+outputVariable);

                System.out.println( TAG+" "+" sameDiff3 - outArray.shape().length 3-  "+ outArray.shape().length);
                System.out.println( TAG+" "+" sameDiff3 - outArray.shape()[0] 3-  "+ outArray.shape()[0]);
                System.out.println( TAG+" "+" sameDiff3 - outArray.shape()[1] 3-  "+ outArray.shape()[1]);
                System.out.println( TAG+" "+" sameDiff3 - outArray.shape()[2] 3-  "+ outArray.shape()[2]);

//                System.out.println( TAG+" "+" sameDiff3 - testData.getLabels().size() - 2 - "+testData.getLabels().size());
                System.out.println( TAG+" "+" sameDiff3 - trainData.totalOutcomes() - 2 - "+trainData.totalOutcomes());
                System.out.println( TAG+" "+" sameDiff3 - placeholderDataDatasets.size() - 2 - "+placeholderDataDatasets.size());
                System.out.println( TAG+" "+" sameDiff3 - placeHolderDataKeyHolder - 2 - "+placeHolderDataKeyHolder);
                System.out.println( TAG+" "+" sameDiff3 - j - 2 - "+j);
//                System.out.println( TAG+" "+" sameDiff3 - placeHolderDataKey - 2 - "+placeHolderDataKey);

                System.out.println( TAG+" "+" sameDiff3 - i 0- "+i);
                System.out.println( TAG+" "+" sameDiff3 - loopCount 0- "+loopCount);
                System.out.println( TAG+" "+" sameDiff3 - placeHolderDatasetArrayList.size() - 1 - "+placeHolderDatasetArrayList.size());

//                lossMSE = sd.loss.meanSquaredError(labelInstance, outReduced, null);
////                lossMSE = sd.loss.meanSquaredError("lossMSE"+" - "+mRandomNumericalId, labelInstance, outReduced, null);
//                System.out.println( TAG+" "+" sameDiff3 - Arrays.toString(lossMSE.getShape()) 4- "+ Arrays.toString(lossMSE.getShape()));
//                System.out.println( TAG+" "+" sameDiff3 - lossMSE.eval().shapeInfoToString() 4- "+ lossMSE.eval().shapeInfoToString());

                History outSingle;
                if(tNext != null)
                {
                    trainData.reset();
                    placeholderData = new HashMap<>();
                    placeholderData.put("input",  trainData.next().getFeatures());
                    placeholderData.put("label", trainData.next().getLabels());
                    trainData.reset();

                    outSingle = sd.fit(tNext);
//                outSingle = sd.fit(t);
                    List<Double> accSingle = outSingle.trainingEval(Evaluation.Metric.ACCURACY);
                    System.out.println( TAG+" "+" sameDiff3 - Accuracy single (tNext): " + accSingle);
                    System.out.println( TAG+" "+" sameDiff3 - outSingle.toString() (tNext): " + outSingle.toString());

                placeholderData = new HashMap<>();
                placeholderData.put("input",  tNext.getFeatures());
                placeholderData.put("label", tNext.getLabels());

                    System.out.println( TAG+" "+" sameDiff3 - outReduced.name() 5-  "+ outReduced.name());
                    INDArray labelArray = label.eval(placeholderData);
                    INDArray outReducedArrayFinal = outReduced.getArr();

                    System.out.println( TAG+" "+" sameDiff3 - ======================================================= - ");
                    System.out.println( TAG+" "+" sameDiff3 - label.eval(placeholderData) 2- "+ labelArray);
                    System.out.println( TAG+" "+" sameDiff3 - outReduced.getArr() 2-  "+ outReducedArrayFinal);

                    System.out.println( TAG+" "+" sameDiff3 - Arrays.toString(labelArray.shape()) 2- "+ Arrays.toString(labelArray.shape()));
                    System.out.println( TAG+" "+" sameDiff3 - Arrays.toString(outReducedArrayFinal.shape()) 2-  "+ Arrays.toString(outReducedArrayFinal.shape()));

                    for(int m = 0; m < labelArray.shape()[0]; ++m)
                    {
                        for(int n = 0; n < labelArray.shape()[1]; ++n)
                        {
                            for(int k = 0; k < labelArray.shape()[2]; ++k)
                            {
                                double mseError = Math.pow((labelArray.getDouble(m, n, k) - outReducedArrayFinal.getDouble(m, n, k)), 2);
                                System.out.println( TAG+" "+" sameDiff3 - mseError 2-  "+ mseError);
                            }
                        }
                    }

                    t2 = tNext;

                    tNext = null;

//                    evaluation = outSingle.finalTrainingEvaluations().evaluation(outReduced);
//
//                    System.out.println(evaluation.stats());
//
//                    float[] losses = outSingle.lossCurve().meanLoss(lossMSE);
//
//                    System.out.println( TAG+" "+" sameDiff3 - Losses: " + Arrays.toString(losses));

                }
                else
                {

                    placeholderData = new HashMap<>();
                    placeholderData.put("input",  t.getFeatures());
                    placeholderData.put("label", t.getLabels());

//                    outSingle = sd.fit(tNext);
                    outSingle = sd.fit(t);
                    List<Double> accSingle = outSingle.trainingEval(Evaluation.Metric.ACCURACY);
                    System.out.println( TAG+" "+" sameDiff3 - Accuracy single (t): " + accSingle);
                    System.out.println( TAG+" "+" sameDiff3 - outSingle.toString() (tNext): " + outSingle.toString());

                    placeholderData = new HashMap<>();
                    placeholderData.put("input",  t.getFeatures());
                    placeholderData.put("label", t.getLabels());

                    System.out.println( TAG+" "+" sameDiff3 - outReduced.name() 6-  "+ outReduced.name());
                    INDArray labelArray = label.eval(placeholderData);
                    INDArray outReducedArrayFinal = outReduced.getArr();

                    System.out.println( TAG+" "+" sameDiff3 - ======================================================= - ");
                    System.out.println( TAG+" "+" sameDiff3 - label.eval(placeholderData) 2- "+ labelArray);
                    System.out.println( TAG+" "+" sameDiff3 - outReduced.getArr() 2-  "+ outReducedArrayFinal);

                    System.out.println( TAG+" "+" sameDiff3 - Arrays.toString(labelArray.shape()) 2- "+ Arrays.toString(labelArray.shape()));
                    System.out.println( TAG+" "+" sameDiff3 - Arrays.toString(outReducedArrayFinal.shape()) 2-  "+ Arrays.toString(outReducedArrayFinal.shape()));

                    for(int m = 0; i < labelArray.shape()[0]; ++m)
                    {
                        for(int n = 0; n < labelArray.shape()[1]; ++n)
                        {
                            for(int k = 0; k < labelArray.shape()[2]; ++k)
                            {
                                double mseError = Math.pow((labelArray.getDouble(m, n, k) - outReducedArrayFinal.getDouble(m, n, k)), 2);
                                System.out.println( TAG+" "+" sameDiff3 - mseError 2-  "+ mseError);
                            }
                        }
                    }

                    t2 = t;

//                    evaluation = outSingle.finalTrainingEvaluations().evaluation(outReduced);
//
//                    System.out.println(evaluation.stats());
//
//                    float[] losses = outSingle.lossCurve().meanLoss(lossMSE);
//
//                    System.out.println( TAG+" "+" sameDiff3 - Losses: " + Arrays.toString(losses));

                }

//                eval.evalTimeSeries(labels,outSingle.getLossCurve().getLossValues());

                //Evaluate on test set:
//                sd.evaluate(testData, outputVariable, eval);

                //Print evaluation statistics:
//                System.out.println( TAG+" "+" sameDiff3 - evaluation.stats() - single - "+eval.stats());

                System.out.println( TAG+" "+" sameDiff3 - i 1- "+i);

////                if(loopCount == 3)
//                    if(loopCount == 2)
////                    if(loopCount == 1)
//                {
//                    System.out.println( TAG+" "+" sameDiff3 - i 1- "+i);
//                    System.out.println( TAG+" "+" sameDiff3 - loopCount 1- "+loopCount);
////                    placeHolderDatasetArrayList.remove(0);
//
////                    t2 = t;
//
//                    loopCount = 0;
//
////                    ++loopCount;
//                    break;
//                }

            }

//            testDataIndex = 0;
//            testData.reset();
//            while(testData.hasNext())
//            {
//                ++testDataIndex;
//                System.out.println( TAG+" "+" sameDiff3 - testDataIndex - " + testDataIndex);
////                System.out.println( TAG+" "+" testData.next() - " + testData.next());
//
//                DataSet t = testData.next();
////                DataSet t = testData.next(testDataIndex - 1);
//                INDArray features = t.getFeatures();
//                INDArray labels = t.getLabels();
//
//                dim0 = t.getFeatures().size(0);
//                dim1 = t.getFeatures().size(1);
//                dim2 = t.getFeatures().size(2);
//                System.out.println( TAG+" "+" sameDiff3 - features - dim0 - 1 - "+dim0);
//                System.out.println( TAG+" "+" sameDiff3 - features - dim1 - 1 - "+dim1);
//                System.out.println( TAG+" "+" sameDiff3 - features - dim2 - 1 - "+dim2);
//
//                placeholderData = new HashMap<>();
//
//                placeholderData.put("input",  t.getFeatures());
//                placeholderData.put("label", t.getLabels());
//
//                System.out.println( TAG+" "+" sameDiff3 - ======================================================= - ");
//                System.out.println( TAG+" "+" sameDiff3 - Arrays.toString(label.eval().shape()) - "+ Arrays.toString(label.eval(placeholderData).shape()));
//
//                outArray = tf_model.forward(encInput, decInput, decSourceMask, decInputMask).getArr();
//                System.out.println( TAG+" "+" sameDiff3 - outArray.shape().length 2-  "+ outArray.shape().length);
//                System.out.println( TAG+" "+" sameDiff3 - outArray.shape()[0] 2-  "+ outArray.shape()[0]);
//                System.out.println( TAG+" "+" sameDiff3 - outArray.shape()[1] 2-  "+ outArray.shape()[1]);
//                System.out.println( TAG+" "+" sameDiff3 - outArray.shape()[2] 2-  "+ outArray.shape()[2]);
//
//                INDArray outReducedArray = outArray.get(NDArrayIndex.all(), NDArrayIndex.interval(0, labels.shape()[1]), NDArrayIndex.interval(0, labels.shape()[2]));
//                System.out.println( TAG+" "+" sameDiff3 - outReducedArray.shape().length 1-  "+ outReducedArray.shape().length);
//                System.out.println( TAG+" "+" sameDiff3 - outReducedArray.shape()[0] 2-  "+ outReducedArray.shape()[0]);
//                System.out.println( TAG+" "+" sameDiff3 - outReducedArray.shape()[1] 2-  "+ outReducedArray.shape()[1]);
//                System.out.println( TAG+" "+" sameDiff3 - outReducedArray.shape()[2] 2-  "+ outReducedArray.shape()[2]);
//
//                outReduced = sd.var("outReduced"+" - "+mRandomNumericalId, outReducedArray);
//                System.out.println( TAG+" "+" sameDiff3 - Arrays.toString(outReduced.getShape()) 4- "+ Arrays.toString(outReduced.getShape()));
//                System.out.println( TAG+" "+" sameDiff3 - outReduced.eval().shapeInfoToString() 4- "+ outReduced.eval().shapeInfoToString());
//
//                dim0 = t.getFeatures().size(0);
//                dim1 = t.getFeatures().size(1);
//                dim2 = t.getFeatures().size(2);
//                System.out.println( TAG+" "+" sameDiff3 - features - dim0 - 2 - "+dim0);
//                System.out.println( TAG+" "+" sameDiff3 - features - dim1 - 2 - "+dim1);
//                System.out.println( TAG+" "+" sameDiff3 - features - dim2 - 2 - "+dim2);
//
//                labelInstance = sd.var("", labels);
////                labelInstance = sd.var("", label.eval(placeholderData));
//                System.out.println( TAG+" "+" sameDiff3 - Arrays.toString(labelInstance.getShape()) 4- "+ Arrays.toString(labelInstance.getShape()));
//                System.out.println( TAG+" "+" sameDiff3 - labelInstance.eval().shapeInfoToString() 4- "+ labelInstance.eval().shapeInfoToString());
//
//                System.out.println( TAG+" "+" sameDiff3 - outArray.shape().length 3-  "+ outArray.shape().length);
//                System.out.println( TAG+" "+" sameDiff3 - outArray.shape()[0] 3-  "+ outArray.shape()[0]);
//                System.out.println( TAG+" "+" sameDiff3 - outArray.shape()[1] 3-  "+ outArray.shape()[1]);
//                System.out.println( TAG+" "+" sameDiff3 - outArray.shape()[2] 3-  "+ outArray.shape()[2]);
//
//                System.out.println( TAG+" "+" sameDiff3 - ======================================================= - ");
//                System.out.println( TAG+" "+" sameDiff3 - Arrays.toString(label.eval().shape()) - "+ Arrays.toString(label.eval(placeholderData).shape()));
//
//                lossMSE = sd.loss.meanSquaredError("lossMSE"+" - "+mRandomNumericalId, labelInstance, outReduced, null);
//                System.out.println( TAG+" "+" sameDiff3 - Arrays.toString(lossMSE.getShape()) 4- "+ Arrays.toString(lossMSE.getShape()));
//                System.out.println( TAG+" "+" sameDiff3 - lossMSE.eval().shapeInfoToString() 4- "+ lossMSE.eval().shapeInfoToString());
//
//
//                History outSingle = sd.fit(t);
//                List<Double> accSingle = outSingle.trainingEval(Evaluation.Metric.ACCURACY);
//                System.out.println( TAG+" "+" sameDiff3 - Accuracy single: " + accSingle);
//
////                eval.evalTimeSeries(labels,outSingle.getLossCurve().getLossValues());
//
//                //Evaluate on test set:
//                sd.evaluate(testData, outputVariable, eval);
//
//                //Print evaluation statistics:
//                System.out.println( TAG+" "+" sameDiff3 - evaluation.stats() - single - "+eval.stats());
//            }

//DISABLE - START - (^)(^)(^)(^)(^)(^)(^)(^)(^)(^)(^)(^)(^)(^)(^)(^)(^)(^)(^)(^)(^)(^)(^)(^)(^)(^)(^)(^)(^)(^)(^)(^)(^)(^)
//            trainData.reset();
//            testData.reset();
//
////            //Evaluate on test set:
////            sd.evaluate(testData, outputVariable, eval);
////
////            //Print evaluation statistics:
////            System.out.println( TAG+" "+" sameDiff3 - evaluation.stats() - "+eval.stats());
//
//            System.out.println( TAG+" "+" sameDiff3 - Calling createAndConfigureModel a second time --- ");
//
//            placeholderData = new HashMap<>();
//            testData.reset();
//            placeholderData.put("input",  testData.next().getFeatures());
//            placeholderData.put("label", testData.next().getLabels());
//            testData.reset();
//
//            createAndConfigureModel(placeholderData);
//
//            System.out.println( TAG+" "+" sameDiff3 - outReduced.name() 7-  "+ outReduced.name());
//            outputVariable = outReduced.name();
//
////            outReduced.setVarName(outReducedName);
////            System.out.println( TAG+" "+" sameDiff3 - outReduced.name() 7a-  "+ outReduced.name());
//
//            fitAndEvaluateTestDataset();
//
//DISABLE - END - (^)(^)(^)(^)(^)(^)(^)(^)(^)(^)(^)(^)(^)(^)(^)(^)(^)(^)(^)(^)(^)(^)(^)(^)(^)(^)(^)(^)(^)(^)(^)(^)(^)(^)

            System.out.println( TAG+" "+" sameDiff3 - Epoch " + i + " ending ");

        }

//        int whileLoopIndex = 0;

//        while(trainData.hasNext()){
//            DataSet t = trainData.next();
//            System.out.println( TAG+" "+" Printing traindata dataset shape");
//            System.out.println(java.util.Arrays.toString(t.getFeatures().shape()));
//            System.out.println( TAG+" "+" whileLoopIndex - "+whileLoopIndex);
//            ++whileLoopIndex;
//        }

        //TRAINING PER DATASET - START - [_][_][_][_][_][_][_][_][_][_][_][_][_][_][_][_][_][_][_][_][_][_][_][_][_][_][_]
//        DataSetIterator trainDataCopy = trainData;
//        whileLoopIndex = -1;
//        trainData.reset();
//        while(trainData.hasNext()) {
//            ++whileLoopIndex;
//            placeholderData = new HashMap<>();
//            t = trainData.next();
////            System.out.println( TAG+" "+"t.toString() - "+ t.toString());
////            System.out.println( TAG+" "+" ======================================================= - ");
////            System.out.println( TAG+" "+" ======================================================= - ");
//            System.out.println( TAG+" "+" ======================================================= - ");
//            System.out.println( TAG+" "+" Printing traindata feature and label dataset shape");
//            System.out.println(Arrays.toString(t.getFeatures().shape()));
//            System.out.println(Arrays.toString(t.getLabels().shape()));
//            System.out.println( TAG+" "+" ======================================================= - ");
////            System.out.println( TAG+" "+" ======================================================= - ");
//
//            INDArray features = t.getFeatures();
//            INDArray labels = t.getLabels();
//            placeholderData.put("input", features);
//            placeholderData.put("label", labels);
//
//            dim0 = t.getFeatures().size(0);
//            dim1 = t.getFeatures().size(1);
//            dim2 = t.getFeatures().size(2);
//
//            System.out.println( TAG+" "+" features - dim0 - "+dim0);
//            System.out.println( TAG+" "+" features - dim1 - "+dim1);
//            System.out.println( TAG+" "+" features - dim2 - "+dim2);
//
////            getConfiguration();
//
////            //Perform training for 2 epochs
////            int numEpochs = 2;
////            sd.fit(trainData, numEpochs);
//
//            System.out.println( TAG+" "+" ======================================================= - ");
//            System.out.println( TAG+" "+" t.getFeatures().size(2) - "+ t.getFeatures().size(2));
//            System.out.println( TAG+" "+" miniBatchSize) - "+ miniBatchSize);
//            System.out.println( TAG+" "+" nOut - "+ nOut);
//
//            History history = sd.fit(t);
////            System.out.println( TAG+" "+" ======================================================= - ");
////            System.out.println( TAG+" "+"history.toString() - "+history.toString());
//
//
//
////            Map<String,INDArray> placeholderData2 = new HashMap<>();
////
////            placeholderData2 = sd.output(placeholderData, "loss");
////
////            INDArray result;
////            for(String placeholderData2Key : placeholderData2.keySet())
////            {
////
////                System.out.println( TAG+" "+" ======================================================= - ");
////                System.out.println( TAG+" "+" placeholderData2.get(placeholderData2Key).shapeInfoToString() - "+placeholderData2.get(placeholderData2Key).shapeInfoToString());
////
////            }
//
//            System.out.println( TAG+" "+" Completed training run --- ");
//
//        }
        //TRAINING PER DATASET - END - [_][_][_][_][_][_][_][_][_][_][_][_][_][_][_][_][_][_][_][_][_][_][_][_][_][_][_]

//        System.out.println( TAG+" "+" Starting test data evaluation --- ");
//
//        //Evaluate on test set:
//        sd.evaluate(testData, outputVariable, eval);
////        sd.evaluate(testData, outputVariable, evaluation);
//
//        //Print evaluation statistics:
//        System.out.println( TAG+" "+" evaluation.stats() - "+eval.stats());
////        System.out.println( TAG+" "+" evaluation.stats() - "+evaluation.stats());

        System.out.println( TAG+" "+"----- Example Complete -----");

        String pathToSavedNormalizer = "src/main/assets/location_next_neural_network_v7_04_normalizer.zip";
        File savedNormalizer = new File(pathToSavedNormalizer);
        NormalizerSerializer mNormalizerSerializer = new NormalizerSerializer();
//        NormalizerSerializer mNormalizerSerializer = NormalizerSerializer.getDefault();
        mNormalizerSerializer.addStrategy( new StandardizeSerializerStrategy() );
        mNormalizerSerializer.write(normalizer, savedNormalizer);

//        String pathToSavedNetwork = "src/main/assets/location_next_neural_network_v7_04.zip";
//        File savedNetwork = new File(pathToSavedNetwork);

//        ModelSerializer.writeModel(restored, savedNetwork,true,savedNetworkNormalizer);
//        ModelSerializer.writeModel(net, savedNetwork,true,savedNetworkNormalizer);
//        ModelSerializer.writeModel(net, savedNetwork,true,normalizer);

//        sd.save(savedNetwork, true);
//        ModelSerializer.addNormalizerToModel(savedNetwork, normalizer);

        //Save the trained network for inference - FlatBuffers format
        File saveFileForInference = new File("src/main/assets/location_next_neural_network_v7_04.fb");
//        File saveFileForInference = new File("/home/adonnini1/Development/ContextQSourceCode/NeuralNetworks/deeplearning4j-examples-master_1/dl4j-examples/src/main/assets/location_next_neural_network_v6_07_04.fb");

        try {
            sd.asFlatFile(saveFileForInference, false);
//            sd.asFlatFile(saveFileForInference, true);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

//        SameDiff loadedForInference = SameDiff.create();
//
//        if(saveFileForInference.exists()) {
//            System.out.println( TAG+" "+" saveFileForInference exists --- ");
//            System.out.println( TAG+" "+" saveFileForInference.getAbsolutePath() - "+saveFileForInference.getAbsolutePath());
//            System.out.println( TAG+" "+" saveFileForInference.length() - "+saveFileForInference.length());
//
//            try {
//                byte[] bytes;
//                try (InputStream is = new BufferedInputStream(new FileInputStream(saveFileForInference))) {
//                    bytes = IOUtils.toByteArray(is);
//                }
//                ByteBuffer bbIn = ByteBuffer.wrap(bytes);
//                System.out.println( TAG+" "+" bbIn.getChar(25) - " + bbIn.getChar(25));
//
////                loadedForInference = SameDiff.fromFlatBuffers(bbIn, true);
//            loadedForInference = SameDiff.fromFlatFile(saveFileForInference);
//            } catch (IOException e) {
//                throw new RuntimeException(e);
//            }
//        }
//        //Perform inference on restored network
//
//        INDArray example;
//        INDArray datasetIteratorForRestoredNetwork;
//
//        //            example = new MnistDataSetIterator(1, false, 12345).next().getFeatures();
//        datasetIteratorForRestoredNetwork = new SequenceRecordReaderDataSetIterator(trainFeatures, trainLabels, miniBatchSize, numLabelClasses,
//                true, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END).next().getFeatures();
//
////        loadedForInference.getVariable("input").setArray(example);
//        loadedForInference.getVariable("input").setArray(datasetIteratorForRestoredNetwork);
//        try (INDArray output = loadedForInference.getVariable("softmax").eval()) {
//
//            System.out.println( TAG+" "+"-----------------------");
////        System.out.println(example.reshape(28, 28));
//            System.out.println( TAG+" "+"Output probabilities: " + output);
//            System.out.println( TAG+" "+"Predicted class: " + output.argMax().getInt(0));
//        }

    }

    private static void fitAndEvaluateTestDataset()
    {
        HashMap<Integer,INDArray> placeholderDataLabels = new HashMap<>();
        HashMap<Integer,INDArray> placeholderDataInputs = new HashMap<>();
        HashMap<Integer,DataSet> placeholderDataDatasets = new HashMap<>();

        System.out.println( TAG+" "+" fitAndEvaluateTestDataset - testData.totalOutcomes() - 0 - "+testData.totalOutcomes());

        int testDataIndex = 0;
        testData.reset();
        while(testData.hasNext())
        {
            System.out.println( TAG+" "+" fitAndEvaluateTestDataset - testDataIndex - placeholderDataInputs - "+testDataIndex);
            placeholderDataInputs.put(testDataIndex,  testData.next().getFeatures());
            testDataIndex++;
//                ++testDataIndex;
        }

        testData.reset();
        testDataIndex = 0;
        while(testData.hasNext())
        {
            System.out.println( TAG+" "+" fitAndEvaluateTestDataset - testDataIndex - placeholderDataLabels - "+testDataIndex);
            DataSet testDataDataSet = testData.next();
            placeholderDataLabels.put(testDataIndex, testDataDataSet.getLabels());
            testDataIndex++;
//                ++testDataIndex;

            long[] testDataDataSetShape = testDataDataSet.getLabels().shape();
            int testDataDim0 = (int) testDataDataSetShape[0];
            int testDataDim1 = (int) testDataDataSetShape[1];
            int testDataDim2 = (int) testDataDataSetShape[2];

            System.out.println( TAG+" "+" fitAndEvaluateTestDataset - testDataDim0 - "+testDataDim0);
            System.out.println( TAG+" "+" fitAndEvaluateTestDataset - testDataDim1 - "+testDataDim1);
            System.out.println( TAG+" "+" fitAndEvaluateTestDataset - testDataDim2 - "+testDataDim2);

        }
        testData.reset();
        testDataIndex = 0;

        while(testData.hasNext())
        {
            System.out.println( TAG+" "+" fitAndEvaluateTestDataset - testDataIndex - placeholderDataDatasets - "+testDataIndex);
            placeholderDataDatasets.put(testDataIndex, testData.next());
            testDataIndex++;
//                ++testDataIndex;
        }
            testData.reset();

//            testDataIndex = 0;
//            while(testData.hasNext())
//            {
//                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - testDataIndex - 0 - "+testDataIndex);
//                placeholderDataInputs.put(testDataIndex,  testData.next().getFeatures());
//                placeholderDataLabels.put(testDataIndex, testData.next().getLabels());
//                placeholderDataDatasets.put(testDataIndex, testData.next());
//                testDataIndex++;
////                ++testDataIndex;
//            }
//            System.out.println( TAG+" "+" fitAndEvaluateTestDataset - placeholderDataInputs.size() - 2 - "+placeholderDataInputs.size());
//            System.out.println( TAG+" "+" fitAndEvaluateTestDataset - placeholderDataLabels.size() - 2 - "+placeholderDataLabels.size());
//            System.out.println( TAG+" "+" fitAndEvaluateTestDataset - placeholderDataDatasets.size() - 2 - "+placeholderDataDatasets.size());

//        testData.reset();

        Integer placeHolderDataKeyHolder = 0;
        int loopCount = 0;

        ArrayList<DataSet> placeHolderDatasetArrayList = new ArrayList<DataSet>();
        int placeHolderDatasetArrayListRecordIndex = 0;
        for(Integer placeHolderDataDatasetsKey : placeholderDataDatasets.keySet())
        {
            placeHolderDatasetArrayList.add(placeholderDataDatasets.get(placeHolderDataDatasetsKey));
        }

//            int recordRemovalIndex = 0;
//            while(recordRemovalIndex < nEpochs && placeHolderDatasetArrayList.size() > 0)
//            {
//                placeHolderDatasetArrayList.remove(0);
//                ++recordRemovalIndex;
//                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - placeHolderDatasetArrayList.size() - 0 - "+placeHolderDatasetArrayList.size());
//            }

        int placeHolderDatasetArrayListSize = placeHolderDatasetArrayList.size();
        System.out.println( TAG+" "+" fitAndEvaluateTestDataset - placeHolderDatasetArrayListSize - 1 - "+placeHolderDatasetArrayListSize);

        for(int j = 0; j < placeHolderDatasetArrayListSize; ++j)
//            for(int j = i; j < placeHolderDatasetArrayListSize; ++j)
//                for(int j = loopCount; j < placeHolderDatasetArrayListSize; ++j)
//            for(int j = placeHolderDatasetArrayListSize - 1; j >= 0; --j)
//                for(Integer placeHolderDataKey : placeholderDataInputs.keySet())
        {

            mRandom = new Random();
            mRandomNumericalId = mRandom.nextInt(100000000);

            NameScope ns = sd.withNameScope("Testing"+" - "+mRandomNumericalId);

            INDArray featuresNext = null;
            INDArray labelsNext = null;
//            DataSet tNext = null;
//            DataSet t = null;

            ++loopCount;

            System.out.println( TAG+" "+" fitAndEvaluateTestDataset - placeHolderDataKeyHolder - 1 - "+placeHolderDataKeyHolder);
            System.out.println( TAG+" "+" fitAndEvaluateTestDataset - j - 1 - "+j);
//                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - placeHolderDataKey - 1 - "+placeHolderDataKey);

            System.out.println( TAG+" "+" fitAndEvaluateTestDataset - (placeHolderDatasetArrayListSize - 1) - 1 - "+(placeHolderDatasetArrayListSize - 1));

            dim0 = t2.getFeatures().size(0);
            dim1 = t2.getFeatures().size(1);
            dim2 = t2.getFeatures().size(2);
            System.out.println( TAG+" "+" fitAndEvaluateTestDataset - t2.getFeatures() - dim0 - t2 - "+dim0);
            System.out.println( TAG+" "+" fitAndEvaluateTestDataset - t2.getFeatures() - dim1 - t2 - "+dim1);
            System.out.println( TAG+" "+" fitAndEvaluateTestDataset - t2.getFeatures() - dim2 - t2 - "+dim2);


            if(((placeHolderDatasetArrayListSize - 1) == 0))
            {
                System.out.println( TAG+" "+" fitAndEvaluateTestDataset -----------((placeHolderDatasetArrayListSize - 1) == 0)------------");

                tNext = placeHolderDatasetArrayList.get(j);
                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - tNext.getFeatures().size(0) - 2 - "+tNext.getFeatures().size(0));
                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - tNext.getFeatures().size(1) - 2 - "+tNext.getFeatures().size(1));
                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - tNext.getFeatures().size(2) - 2 - "+tNext.getFeatures().size(2));

                t = placeholderDataDatasets.get(j);
//                INDArray features = placeholderDataInputs.get(placeHolderDataKey);
//                INDArray labels = placeholderDataLabels.get(placeHolderDataKey);
//                DataSet t = placeholderDataDatasets.get(placeHolderDataKey);

//            dim0 = t2.getFeatures().size(0);
//            dim1 = t2.getFeatures().size(1);
//            dim2 = t2.getFeatures().size(2);
//            System.out.println( TAG+" "+" fitAndEvaluateTestDataset - features - dim0 - t2 - "+dim0);
//            System.out.println( TAG+" "+" fitAndEvaluateTestDataset - features - dim1 - t2 - "+dim1);
//            System.out.println( TAG+" "+" fitAndEvaluateTestDataset - features - dim2 - t2 - "+dim2);

                dim0 = t.getFeatures().size(0);
                dim1 = t.getFeatures().size(1);
                dim2 = t.getFeatures().size(2);
                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - t.getFeatures() - dim0 - 1 - "+dim0);
                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - t.getFeatures() - dim1 - 1 - "+dim1);
                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - t.getFeatures() - dim2 - 1 - "+dim2);

            }
            if(((placeHolderDatasetArrayListSize - 1) > 0) && (j == 0))
            {
                System.out.println( TAG+" "+" fitAndEvaluateTestDataset -----------((placeHolderDatasetArrayListSize - 1) > 0) && (j == 0)------------");

//                tNext = placeHolderDatasetArrayList.get(j);
                tNext = placeHolderDatasetArrayList.get(j+1);
                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - tNext.getFeatures().size(0) - 1 - "+tNext.getFeatures().size(0));
                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - tNext.getFeatures().size(1) - 1 - "+tNext.getFeatures().size(1));
                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - tNext.getFeatures().size(2) - 1 - "+tNext.getFeatures().size(2));

                t = placeholderDataDatasets.get(j);
//                INDArray features = placeholderDataInputs.get(placeHolderDataKey);
//                INDArray labels = placeholderDataLabels.get(placeHolderDataKey);
//                DataSet t = placeholderDataDatasets.get(placeHolderDataKey);

//            dim0 = t2.getFeatures().size(0);
//            dim1 = t2.getFeatures().size(1);
//            dim2 = t2.getFeatures().size(2);
//            System.out.println( TAG+" "+" fitAndEvaluateTestDataset - features - dim0 - t2 - "+dim0);
//            System.out.println( TAG+" "+" fitAndEvaluateTestDataset - features - dim1 - t2 - "+dim1);
//            System.out.println( TAG+" "+" fitAndEvaluateTestDataset - features - dim2 - t2 - "+dim2);

                dim0 = t.getFeatures().size(0);
                dim1 = t.getFeatures().size(1);
                dim2 = t.getFeatures().size(2);
                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - features - dim0 - 1 - "+dim0);
                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - features - dim1 - 1 - "+dim1);
                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - features - dim2 - 1 - "+dim2);

            }
            else if(((placeHolderDatasetArrayListSize - 1) > 0) && (j == (placeHolderDatasetArrayListSize - 1)))
            {
                System.out.println( TAG+" "+" fitAndEvaluateTestDataset -----------((placeHolderDatasetArrayListSize - 1) > 0) && (j == (placeHolderDatasetArrayListSize - 1))------------");

                tNext = placeHolderDatasetArrayList.get(j);
                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - tNext.getFeatures().size(0) - 2 - "+tNext.getFeatures().size(0));
                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - tNext.getFeatures().size(1) - 2 - "+tNext.getFeatures().size(1));
                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - tNext.getFeatures().size(2) - 2 - "+tNext.getFeatures().size(2));

            }
            else if(((placeHolderDatasetArrayListSize - 1) > 0) && (j < (placeHolderDatasetArrayListSize - 1)))
            {
                System.out.println( TAG+" "+" fitAndEvaluateTestDataset -----------((placeHolderDatasetArrayListSize - 1) > 0) && (j < (placeHolderDatasetArrayListSize - 1))------------");

                tNext = placeHolderDatasetArrayList.get(j);
//                tNext = placeHolderDatasetArrayList.get(j+1);
                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - tNext.getFeatures().size(0) - 3 - "+tNext.getFeatures().size(0));
                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - tNext.getFeatures().size(1) - 3 - "+tNext.getFeatures().size(1));
                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - tNext.getFeatures().size(2) - 3 - "+tNext.getFeatures().size(2));

                t = placeholderDataDatasets.get(j);

                dim0 = t.getFeatures().size(0);
                dim1 = t.getFeatures().size(1);
                dim2 = t.getFeatures().size(2);
                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - features - dim0 - 3 - "+dim0);
                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - features - dim1 - 3 - "+dim1);
                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - features - dim2 - 3 - "+dim2);

                INDArray outReducedArray = outArray.get(NDArrayIndex.interval(0, tNext.getLabels().size(0)), NDArrayIndex.interval(0, tNext.getLabels().size(1)), NDArrayIndex.interval(0, tNext.getLabels().size(2)));
//                    INDArray outReducedArray = outArray.get(NDArrayIndex.interval(0, tNext.getFeatures().size(0)), NDArrayIndex.interval(0, tNext.getFeatures().size(1)), NDArrayIndex.interval(0, tNext.getFeatures().size(2)));
                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - outReducedArray.shape().length 3-  "+ outReducedArray.shape().length);
                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - outReducedArray.shape()[0] 3-  "+ outReducedArray.shape()[0]);
                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - outReducedArray.shape()[1] 3-  "+ outReducedArray.shape()[1]);
                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - outReducedArray.shape()[2] 3-  "+ outReducedArray.shape()[2]);
                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - outReducedArray 3-  "+ outReducedArray);

                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - outReduced.name() 8-  "+ outReduced.name());
                sd.associateArrayWithVariable(outReducedArray, outReduced);
//                    outReduced = sd.var("outReduced"+" - "+mRandomNumericalId, outReducedArray);
                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - outReduced.getShape()[0] 3-  "+ outReduced.getShape()[0]);
                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - outReduced.getShape()[1] 3-  "+ outReduced.getShape()[1]);
                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - outReduced.getShape()[2] 3-  "+ outReduced.getShape()[2]);
                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - outReduced.getArr() 3-  "+ outReduced.getArr());

            }

//fitAndEvaluateTestDataset DATASET tNext  - START - _+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+
//            if(j < placeHolderDatasetArrayListSize - 1)
////                if(placeHolderDataKeyHolder.equals(j))
////                    if(placeHolderDataKeyHolder.equals(placeHolderDataKey))
//            {
//                {
//                    featuresNext = placeholderDataInputs.get(j + 1);
//                    labelsNext = placeholderDataLabels.get(j + 1);
//                    tNext = placeHolderDatasetArrayList.get(j + 1);
////                    featuresNext = placeholderDataInputs.get(j - 1);
////                    labelsNext = placeholderDataLabels.get(j - 1);
////                    tNext = placeHolderDatasetArrayList.get(j - 1);
////                    tNext = placeholderDataDatasets.get(j + 1);
////                    featuresNext = placeholderDataInputs.get(placeHolderDataKey + 1);
////                    labelsNext = placeholderDataLabels.get(placeHolderDataKey + 1);
////                    tNext = placeholderDataDatasets.get(placeHolderDataKey + 1);
//                }
//                ++placeHolderDataKeyHolder;
//                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - tNext.getFeatures().size(0) - 1 - "+tNext.getFeatures().size(0));
//                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - tNext.getFeatures().size(1) - 1 - "+tNext.getFeatures().size(1));
//                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - tNext.getFeatures().size(2) - 1 - "+tNext.getFeatures().size(2));
//
//            }
//            else
//            {
//                featuresNext = placeholderDataInputs.get(j);
//                labelsNext = placeholderDataLabels.get(j);
//                tNext = placeHolderDatasetArrayList.get(j);
//                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - tNext.getFeatures().size(0) - 2 - "+tNext.getFeatures().size(0));
//                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - tNext.getFeatures().size(1) - 2 - "+tNext.getFeatures().size(1));
//                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - tNext.getFeatures().size(2) - 2 - "+tNext.getFeatures().size(2));
//            }
//fitAndEvaluateTestDataset DATASET tNext  - END - _+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+

//fitAndEvaluateTestDataset DATASET t  - START - =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
//                INDArray features = placeholderDataInputs.get(j);
//            INDArray labels = placeholderDataLabels.get(j);
//            DataSet t = placeholderDataDatasets.get(j);
////                INDArray features = placeholderDataInputs.get(placeHolderDataKey);
////                INDArray labels = placeholderDataLabels.get(placeHolderDataKey);
////                DataSet t = placeholderDataDatasets.get(placeHolderDataKey);
//
////            dim0 = t2.getFeatures().size(0);
////            dim1 = t2.getFeatures().size(1);
////            dim2 = t2.getFeatures().size(2);
////            System.out.println( TAG+" "+" fitAndEvaluateTestDataset - features - dim0 - t2 - "+dim0);
////            System.out.println( TAG+" "+" fitAndEvaluateTestDataset - features - dim1 - t2 - "+dim1);
////            System.out.println( TAG+" "+" fitAndEvaluateTestDataset - features - dim2 - t2 - "+dim2);
//
//            dim0 = t.getFeatures().size(0);
//            dim1 = t.getFeatures().size(1);
//            dim2 = t.getFeatures().size(2);
//            System.out.println( TAG+" "+" fitAndEvaluateTestDataset - features - dim0 - 1 - "+dim0);
//            System.out.println( TAG+" "+" fitAndEvaluateTestDataset - features - dim1 - 1 - "+dim1);
//            System.out.println( TAG+" "+" fitAndEvaluateTestDataset - features - dim2 - 1 - "+dim2);
//
////                placeholderData = new HashMap<>();
////                placeholderData.put("input",  t.getFeatures());
////                placeholderData.put("label", t.getLabels());
////                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - ======================================================= - ");
////                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - Arrays.toString(label.eval().shape()) - "+ Arrays.toString(label.eval(placeholderData).shape()));
//
////                createAndConfigureModel();
//
////                outArray = tf_model.forward(encInput, decInput, decSourceMask, decInputMask).getArr();
////                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - outArray.shape().length 2-  "+ outArray.shape().length);
////                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - outArray.shape()[0] 2-  "+ outArray.shape()[0]);
////                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - outArray.shape()[1] 2-  "+ outArray.shape()[1]);
////                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - outArray.shape()[2] 2-  "+ outArray.shape()[2]);
//
////            dim0 = t.getLabels().size(0);
////            dim1 = t.getLabels().size(1);
////            dim2 = t.getLabels().size(2);
////            System.out.println( TAG+" "+" fitAndEvaluateTestDataset - labels - dim0 - 2 - "+dim0);
////            System.out.println( TAG+" "+" fitAndEvaluateTestDataset - labels - dim1 - 2 - "+dim1);
////            System.out.println( TAG+" "+" fitAndEvaluateTestDataset - labels - dim2 - 2 - "+dim2);
//fitAndEvaluateTestDataset DATASET t  - END - =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

                INDArray outReducedArray = outArray.get(NDArrayIndex.all(), NDArrayIndex.interval(0, dim1), NDArrayIndex.interval(0, dim2));
//                INDArray outReducedArray = outArray.get(NDArrayIndex.all(), NDArrayIndex.interval(0, labels.shape()[1]), NDArrayIndex.interval(0, labels.shape()[2]));
                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - outReducedArray.shape().length 1-  "+ outReducedArray.shape().length);
                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - outReducedArray.shape()[0] 2-  "+ outReducedArray.shape()[0]);
                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - outReducedArray.shape()[1] 2-  "+ outReducedArray.shape()[1]);
                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - outReducedArray.shape()[2] 2-  "+ outReducedArray.shape()[2]);

//                outReduced = sd.var(outReducedArray);
////                outReduced = sd.var("outReduced2"+" - "+mRandomNumericalId, outReducedArray);
////                outReduced = sd.var("outReduced"+" - "+mRandomNumericalId, outReducedArray);
//                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - Arrays.toString(outReduced.getShape()) 4- "+ Arrays.toString(outReduced.getShape()));
//                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - outReduced.eval().shapeInfoToString() 4- "+ outReduced.eval().shapeInfoToString());
//
//                labelInstance = sd.var("labelInstance"+" - "+mRandomNumericalId, t.getLabels());
//                labelInstance = sd.var(t.getLabels());
//                labelInstance = sd.var("", labels);
//                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - Arrays.toString(labelInstance.getShape()) 4- "+ Arrays.toString(labelInstance.getShape()));
//                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - labelInstance.eval().shapeInfoToString() 4- "+ labelInstance.eval().shapeInfoToString());
//
//            System.out.println( TAG+" "+" fitAndEvaluateTestDataset - outArray.shape().length 3-  "+ outArray.shape().length);
//            System.out.println( TAG+" "+" fitAndEvaluateTestDataset - outArray.shape()[0] 3-  "+ outArray.shape()[0]);
//            System.out.println( TAG+" "+" fitAndEvaluateTestDataset - outArray.shape()[1] 3-  "+ outArray.shape()[1]);
//            System.out.println( TAG+" "+" fitAndEvaluateTestDataset - outArray.shape()[2] 3-  "+ outArray.shape()[2]);

//                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - estData.getLabels().size() - 2 - "+testData.getLabels().size());
            System.out.println( TAG+" "+" fitAndEvaluateTestDataset - testData.totalOutcomes() - 2 - "+testData.totalOutcomes());
            System.out.println( TAG+" "+" fitAndEvaluateTestDataset - placeholderDataDatasets.size() - 2 - "+placeholderDataDatasets.size());
            System.out.println( TAG+" "+" fitAndEvaluateTestDataset - placeHolderDataKeyHolder - 2 - "+placeHolderDataKeyHolder);
            System.out.println( TAG+" "+" fitAndEvaluateTestDataset - j - 2 - "+j);
//                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - placeHolderDataKey - 2 - "+placeHolderDataKey);

//            System.out.println( TAG+" "+" fitAndEvaluateTestDataset - i 0- "+i);
            System.out.println( TAG+" "+" fitAndEvaluateTestDataset - loopCount 0- "+loopCount);
            System.out.println( TAG+" "+" fitAndEvaluateTestDataset - placeHolderDatasetArrayList.size() - 1 - "+placeHolderDatasetArrayList.size());

//                lossMSE = sd.loss.meanSquaredError(labelInstance, outReduced, null);
////                lossMSE = sd.loss.meanSquaredError("lossMSE"+" - "+mRandomNumericalId, labelInstance, outReduced, null);
//                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - Arrays.toString(lossMSE.getShape()) 4- "+ Arrays.toString(lossMSE.getShape()));
//                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - lossMSE.eval().shapeInfoToString() 4- "+ lossMSE.eval().shapeInfoToString());

            History outSingle = null;
            if(tNext != null)
            {
                if(outReducedArray.shape()[0] != tNext.getFeatures().size(0))
                {
                    System.out.println( TAG+" "+" fitAndEvaluateTestDataset - outReducedArray.shape()[0] 3-  "+ outReducedArray.shape()[0]);
                    System.out.println( TAG+" "+" fitAndEvaluateTestDataset - tNext.getFeatures().size(0) 3-  "+ tNext.getFeatures().size(0));

                    INDArray tNextFeaturesResizedArray = Nd4j.create(outReducedArray.shape()[0], outReducedArray.shape()[1], outReducedArray.shape()[2]);
                    INDArray tNextFeaturesResizedPopulatedArray = tNextFeaturesResizedArray.assign(tNext.getFeatures());

                    INDArray tNextLabelsResizedArray = Nd4j.create(outReducedArray.shape()[0], nOut, outReducedArray.shape()[2]);
                    INDArray tNextLabelsResizedPopulatedArray = tNextLabelsResizedArray.assign(tNext.getLabels());


                    if(t2.getFeatures().shape()[2] != tNextLabelsResizedPopulatedArray.shape()[2])
                    {
                        if(t2.getFeatures().shape()[2] > tNextLabelsResizedPopulatedArray.shape()[2])
                        {
                            System.out.println( TAG+" "+" fitAndEvaluateTestDataset - t2.getFeatures().shape()[2] >-  "+ t2.getFeatures().shape()[2]);
                            System.out.println( TAG+" "+" fitAndEvaluateTestDataset - tNextLabelsResizedPopulatedArray.shape()[2] <-  "+ tNextLabelsResizedPopulatedArray.shape()[2]);

                            INDArray testDatasetFeaturesResizedArray = Nd4j.create(t2.getFeatures().shape()[0], t2.getFeatures().shape()[1], t2.getFeatures().shape()[2]);
                            INDArray testDatasetFeaturesResizedPopulatedArray = testDatasetFeaturesResizedArray.assign(tNextFeaturesResizedPopulatedArray);

                            INDArray testDatasetLabelsResizedArray = Nd4j.create(t2.getFeatures().shape()[0], nOut, t2.getFeatures().shape()[2]);
                            INDArray testDatasetLabelsResizedPopulatedArray = testDatasetLabelsResizedArray.assign(tNextLabelsResizedPopulatedArray);

                            DataSet tNextResizedDataset = new DataSet(testDatasetFeaturesResizedPopulatedArray, testDatasetLabelsResizedPopulatedArray);
                            outSingle = sd.fit(tNextResizedDataset);

                        }
                        if(t2.getFeatures().shape()[2] < tNextLabelsResizedPopulatedArray.shape()[2])
                        {
                            System.out.println( TAG+" "+" fitAndEvaluateTestDataset - t2.getFeatures().shape()[2] <-  "+ t2.getFeatures().shape()[2]);
                            System.out.println( TAG+" "+" fitAndEvaluateTestDataset - tNextLabelsResizedPopulatedArray.shape()[2] >-  "+ tNextLabelsResizedPopulatedArray.shape()[2]);

                            INDArray testDatasetFeaturesReducedArray = tNextFeaturesResizedPopulatedArray.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(0, t2.getFeatures().shape()[2]));
                            INDArray testDatasetLabelsReducedArray = tNextLabelsResizedPopulatedArray.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(0, t2.getFeatures().shape()[2]));

                            DataSet testDatasetReducedDataset = new DataSet(testDatasetFeaturesReducedArray, testDatasetLabelsReducedArray);

                            placeholderData = new HashMap<>();

                            placeholderData.put("input",  testDatasetReducedDataset.getFeatures());
                            placeholderData.put("label", testDatasetReducedDataset.getLabels());

                            outSingle = sd.fit(testDatasetReducedDataset);

                        }
                    }

                }
                else
                {

                    System.out.println(TAG+" "+" fitAndEvaluateTestDataset - (outReducedArray.shape()[0] != tNext.getFeatures().size(0)) - " + false);

                    placeholderData = new HashMap<>();

                    placeholderData.put("input",  tNext.getFeatures());
                    placeholderData.put("label", tNext.getLabels());

                    System.out.println( TAG+" "+" fitAndEvaluateTestDataset - outReducedName -  "+ outReducedName);
                    System.out.println( TAG+" "+" fitAndEvaluateTestDataset - outReduced.name() -  "+ outReduced.name());

//                    outReduced.setVarName("");
                    tNext = placeHolderDatasetArrayList.get(j);
                    System.out.println( TAG+" "+" fitAndEvaluateTestDataset - tNext.getFeatures().size(0) 4-  "+ tNext.getFeatures().size(0));

                    try {
                        outSingle = sd.fit(tNext);
                    } catch (Exception e) {
                        System.out.println( TAG+" "+" sd.fit failed ---  ");
                        throw new RuntimeException(e);
                    }

//                outSingle = sd.fit(t2);

                }
//                outSingle = sd.fit(t);
                List<Double> accSingle = outSingle.trainingEval(Evaluation.Metric.ACCURACY);
                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - Accuracy single: " + accSingle);

                outputVariable = outReduced.name();
//        String outputVariable = "out";
//            String outputVariable = "softmax";
                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - outReduced.name() 9-  "+ outReduced.name());
                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - outputVariable 1- " + outputVariable);

                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - lossMSE.name() 1- " + lossMSE.name());
                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - Arrays.toString(lossMSE.getShape()) 4- "+ Arrays.toString(lossMSE.getShape()));
                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - lossMSE.eval(placeholderData).shapeInfoToString() 4- "+ lossMSE.eval(placeholderData).shapeInfoToString());
                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - lossMSE.eval(placeholderData) 4- "+ lossMSE.eval(placeholderData));
//                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - lossMSE.eval().shapeInfoToString() 4- "+ lossMSE.eval().shapeInfoToString());
//                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - lossMSE.eval() - "+ lossMSE.eval());
//        System.out.println( TAG+" "+" fitAndEvaluateTestDataset - lossMSE.eval(placeholderData) - "+ lossMSE.eval(placeholderData));
                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - Arrays.toString(lossMSE.getArr().shape()) 4- "+ Arrays.toString(lossMSE.getArr().shape()));

                placeholderData = new HashMap<>();
                placeholderData.put("input",  tNext.getFeatures());
                placeholderData.put("label", tNext.getLabels());

                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - outReduced.name() 10-  "+ outReduced.name());
                INDArray labelArray = label.eval(placeholderData);
                INDArray outReducedArrayFinal = outReduced.getArr();

                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - ======================================================= - ");
                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - label.eval(placeholderData) 4- "+ labelArray);
                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - outReduced.getArr() 4-  "+ outReducedArrayFinal);

                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - Arrays.toString(labelArray.shape()) 4- "+ Arrays.toString(labelArray.shape()));
                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - Arrays.toString(outReducedArrayFinal.shape()) 4-  "+ Arrays.toString(outReducedArrayFinal.shape()));

                for(int i = 0; i < Math.min(labelArray.shape()[0], outReducedArrayFinal.shape()[0]); ++i)
                {
                    for(int n = 0; n < Math.min(labelArray.shape()[1], outReducedArrayFinal.shape()[1]); ++n)
                    {
                        for(int k = 0; k < Math.min(labelArray.shape()[2], outReducedArrayFinal.shape()[2]); ++k)
                        {
                            double mseError = Math.pow((labelArray.getDouble(i, n, k) - outReducedArrayFinal.getDouble(i, n, k)), 2);
                            System.out.println( TAG+" "+" fitAndEvaluateTestDataset - mseError 4-  "+ mseError);
                        }
                    }
                }

                tNext = null;

////                evaluation = new Evaluation();
////                RegressionEvaluation evaluation = new RegressionEvaluation();
//                RegressionEvaluation evaluation = new RegressionEvaluation(2);
//
//                testData.reset();
//                sd.evaluate(testData, outputVariable, evaluation);
////                sd.evaluate(testData, outputVariable, eval);

            }
            else
            {

                System.out.println(TAG+" "+" fitAndEvaluateTestDataset - (tNext != null) - " + false);

                placeholderData = new HashMap<>();

                placeholderData.put("input",  t.getFeatures());
                placeholderData.put("label", t.getLabels());

                outSingle = sd.fit(t);
//                outSingle = sd.fit(tNext);
//                outSingle = sd.fit(t2);
                List<Double> accSingle = outSingle.trainingEval(Evaluation.Metric.ACCURACY);
                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - Accuracy single: " + accSingle);

                outputVariable = outReduced.name();
//        String outputVariable = "out";
//            String outputVariable = "softmax";
                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - outputVariable 1- " + outputVariable);

                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - lossMSE.name() 1- " + lossMSE.name());
                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - Arrays.toString(lossMSE.getShape()) 4- "+ Arrays.toString(lossMSE.getShape()));
                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - lossMSE.eval().shapeInfoToString() 4- "+ lossMSE.eval().shapeInfoToString());
                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - lossMSE.eval(placeholderData).shapeInfoToString() 4- "+ lossMSE.eval(placeholderData).shapeInfoToString());
//                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - lossMSE.eval() - "+ lossMSE.eval());
                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - lossMSE.eval(placeholderData) - "+ lossMSE.eval(placeholderData));
                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - Arrays.toString(lossMSE.getArr().shape()) 4- "+ Arrays.toString(lossMSE.getArr().shape()));

                placeholderData = new HashMap<>();
                placeholderData.put("input",  t.getFeatures());
                placeholderData.put("label", t.getLabels());

                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - outReduced.name() 11-  "+ outReduced.name());
                INDArray labelArray = label.eval(placeholderData);
                INDArray outReducedArrayFinal = outReduced.getArr();

                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - ======================================================= - ");
                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - label.eval(placeholderData) 4- "+ labelArray);
                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - outReduced.getArr() 4-  "+ outReducedArrayFinal);

                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - Arrays.toString(labelArray.shape()) 4- "+ Arrays.toString(labelArray.shape()));
                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - Arrays.toString(outReducedArrayFinal.shape()) 4-  "+ Arrays.toString(outReducedArrayFinal.shape()));

                for(int i = 0; i < Math.min(labelArray.shape()[0], outReducedArrayFinal.shape()[0]); ++i)
                {
                    for(int n = 0; n < Math.min(labelArray.shape()[1], outReducedArrayFinal.shape()[1]); ++n)
                    {
                        for(int k = 0; k < Math.min(labelArray.shape()[2], outReducedArrayFinal.shape()[2]); ++k)
                        {
                            double mseError = Math.pow((labelArray.getDouble(i, n, k) - outReducedArrayFinal.getDouble(i, n, k)), 2);
                            System.out.println( TAG+" "+" fitAndEvaluateTestDataset - mseError 4-  "+ mseError);
                        }
                    }
                }

////                evaluation = new Evaluation();
////                RegressionEvaluation evaluation = new RegressionEvaluation();
//                RegressionEvaluation evaluation = new RegressionEvaluation(2);
//
//                testData.reset();
//                sd.evaluate(testData, outputVariable, evaluation);
////                sd.evaluate(testData, outputVariable, eval);

            }

//                eval.evalTimeSeries(labels,outSingle.getLossCurve().getLossValues());

            //Evaluate on test set:
//                sd.evaluate(testData, outputVariable, eval);

            //Print evaluation statistics:
//                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - evaluation.stats() - single - "+eval.stats());


////            if(loopCount == 2)
//                    if(loopCount == 1)
//            {
////                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - i 1- "+i);
//                System.out.println( TAG+" "+" fitAndEvaluateTestDataset - loopCount 1- "+loopCount);
////                    placeHolderDatasetArrayList.remove(0);
//                loopCount = 0;
//
////                    ++loopCount;
//                break;
//            }

        }

//                        evaluation = new Evaluation();
        RegressionEvaluation evaluation = new RegressionEvaluation(2);
//                RegressionEvaluation evaluation = new RegressionEvaluation();

//        testData.reset();
//        while(testData.hasNext()) {
////            System.out.println( TAG+" "+" fitAndEvaluateTestDataset - Arrays.toString(testData.next().getLabels().shape()) 4- "+ Arrays.toString(testData.next().getLabels().shape()));
//            evaluation.eval(testData.next().getLabels(), outReduced.getArr());
//            System.out.println( TAG+" "+" fitAndEvaluateTestDataset - evaluation.stats() - "+evaluation.stats());
//        }

        testData.reset();

//        sd.evaluate(testData, outputVariable, evaluation);
////                sd.evaluate(testData, outputVariable, eval);
//
//        //Print evaluation statistics:
//        System.out.println( TAG+" "+" fitAndEvaluateTestDataset - evaluation.stats() - "+evaluation.stats());

    }

    private static void createAndConfigureModel(HashMap<String,INDArray> placeholderData1)
    {
        Nd4j.getExecutioner().enableVerboseMode(true);
        Nd4j.getExecutioner().enableDebugMode(true);

        mRandom = new Random();
        mRandomNumericalId = mRandom.nextInt(100000000);

        NameScope ns = sd.withNameScope("createAndConfigureModel"+" - "+mRandomNumericalId);

//             creating model
//        INPUT:
//        encoder_ip_size - (int) dimension of the encoder input
//        decoder_ip_size - (int) dimension of the decoder input
//        model_op_size - (int) dimension of model's output
//        emb_size - (int) data embedding size for encoder and decoder
//        num_heads - (int) number of heads in multi head attention layer
//        ff_hidden_size - (int) size of the hidden layer for the feed forward network
//        n - (int) number of encoder layers
//        dropout - (float) dropout percentage. Default value = 0.1

//            defining batch size
//        batch_size = 256;
//        batch_size = emb_size / ((int) (emb_size / (lastTestCount)));
//        batch_size = emb_size / ((int) (emb_size / (lastTestCount/2)) - 1);
        batch_size = 128;
//        batch_size = 320;
//        batch_size = 32;
//        int batch_size = 64;

        encoder_ip_size = 6;
//        int encoder_ip_size = 2;
        decoder_ip_size = 6;
//        decoder_ip_size = 3;
        model_op_size = 2;
//        model_op_size = 6;
//        int model_op_size = 3;
        emb_size = 512;
        num_heads = 8;
        ff_hidden_size = 2048;
        n = 1;
//        int n = 6;
        double dropout=0.1;

        labelCount = 2;

        placeholderData = placeholderData1;
//        placeholderData = new HashMap<>();
//        trainData.reset();
//        placeholderData.put("input",  trainData.next().getFeatures());
//        placeholderData.put("label", trainData.next().getLabels());
//        trainData.reset();

//        sd = TransformerArchitectureModel.sd;

        input = sd.placeHolder("input", DataType.FLOAT, -1, encoder_ip_size, -1);
        label = sd.placeHolder("label", DataType.FLOAT, -1, nOut, -1);
//        label = sd.placeHolder("label", DataType.FLOAT, -1, model_op_size, -1);
//        SDVariable input = sd.placeHolder("input", DataType.DOUBLE, -1, encoder_ip_size, -1);
//        SDVariable label = sd.placeHolder("label", DataType.DOUBLE, -1, model_op_size, -1);

        weights = sd.var("weights"+" - "+mRandomNumericalId, new XavierInitScheme('c', encoder_ip_size, model_op_size), DataType.FLOAT, batch_size, encoder_ip_size, labelCount);
//        weights = sd.var("weights"+" - "+mRandomNumericalId, new XavierInitScheme('c', encoder_ip_size, model_op_size), DataType.FLOAT, batch_size, model_op_size, labelCount);
//        SDVariable weights = sd.var("weights", new XavierInitScheme('c', encoder_ip_size, model_op_size), DataType.DOUBLE, batch_size, model_op_size, labelCount);
        bias = sd.constant("bias"+" - "+mRandomNumericalId, 0.0);
//        bias = sd.constant("bias"+" - "+mRandomNumericalId, 0.05);
//        bias = sd.var("bias"+" - "+mRandomNumericalId, Nd4j.rand(DataType.FLOAT, 4 * nOut))
//        SDVariable weights = new SDVariable();
//        SDVariable  bias = new SDVariable();

        System.out.println( TAG+" "+" createAndConfigureModel - Arrays.toString(weights.getShape()) 0- "+ Arrays.toString(weights.getShape()));
        System.out.println( TAG+" "+" createAndConfigureModel - weights.eval().shapeInfoToString() 0- "+ weights.eval().shapeInfoToString());

        tf_model = new TransformerArchitectureModel.TFModel(sd, encoder_ip_size, decoder_ip_size, model_op_size, emb_size,
//                TransformerArchitectureModel.TFModel tf_model = new TransformerArchitectureModel.TFModel(sd, encoder_ip_size, decoder_ip_size, model_op_size, emb_size,
                num_heads, ff_hidden_size, n, dropout, weights, bias, batch_size, labelCount);
        System.out.println( TAG+" "+" createAndConfigureModel - -----------Instantiated TransformerArchitectureModel.TFModel------------");

//        NameScope ns = sd.withNameScope("LossAndConfig");

//        MeanAndStdCalculator();
//        MeanAndStandardDeviationOfTrainDataSet();

//        trainData.reset();
        getEncoderInputDecoderInputAndDecoderMasks();

        System.out.println( TAG+" "+" createAndConfigureModel - Arrays.toString(weights.getShape()) - "+ Arrays.toString(weights.getShape()));
        System.out.println( TAG+" "+" createAndConfigureModel - weights.eval().shapeInfoToString() - "+ weights.eval().shapeInfoToString());
        System.out.println( TAG+" "+" createAndConfigureModel - weights.getArr() - "+ weights.getArr());

        outArray = tf_model.forward(encInput, decInput, decSourceMask, decInputMask).getArr();
        System.out.println( TAG+" "+" createAndConfigureModel - outArray.shape().length 1-  "+ outArray.shape().length);
        System.out.println( TAG+" "+" createAndConfigureModel - outArray.shape()[0] 1-  "+ outArray.shape()[0]);
        System.out.println( TAG+" "+" createAndConfigureModel - outArray.shape()[1] 1-  "+ outArray.shape()[1]);
        System.out.println( TAG+" "+" createAndConfigureModel - outArray.shape()[2] 1-  "+ outArray.shape()[2]);
        System.out.println( TAG+" "+" createAndConfigureModel - outArray 1-  "+ outArray);

        out = sd.var("out"+" - "+mRandomNumericalId, outArray);
//        out = tf_model.forward(encInput, decInput, decSourceMask, decInputMask);
//        out.setVarName("out");
        System.out.println( TAG+" "+" createAndConfigureModel - Arrays.toString(out.getShape()) - "+ Arrays.toString(out.getShape()));
        System.out.println( TAG+" "+" createAndConfigureModel - out.eval().shapeInfoToString() - "+ out.eval().shapeInfoToString());

        System.out.println( TAG+" "+" createAndConfigureModel - placeholderData.get(\"label\").shape()[0] 1-  "+ placeholderData.get("label").shape()[0]);
        System.out.println( TAG+" "+" createAndConfigureModel - placeholderData.get(\"label\").shape()[1] 1-  "+ placeholderData.get("label").shape()[1]);
        System.out.println( TAG+" "+" createAndConfigureModel - placeholderData.get(\"label\").shape()[2] 1-  "+ placeholderData.get("label").shape()[2]);

        System.out.println( TAG+" "+" createAndConfigureModel - ======================================================= - ");
        System.out.println( TAG+" "+" createAndConfigureModel - Arrays.toString(label.eval(placeholderData).shape()) - "+ Arrays.toString(label.eval(placeholderData).shape()));

//        INDArray labelArray = label.eval(placeholderData);
////        INDArray labelArray = label.getArr();
//        INDArray labelArrayResized = Nd4j.create(out.eval().shape()[0], out.eval().shape()[1], out.eval().shape()[2]);
//        INDArray labelArrayResizedPopulated = labelArrayResized.assign(labelArray);
//        labelResizedPopulated = sd.var("labelResizedPopulated"+" - "+mRandomNumericalId, labelArrayResizedPopulated);

        System.out.println( TAG+" "+" createAndConfigureModel - label.eval(placeholderData).shape()[0] 1-  "+ label.eval(placeholderData).shape()[0]);
        System.out.println( TAG+" "+" createAndConfigureModel - label.eval(placeholderData).shape()[1] 1-  "+ label.eval(placeholderData).shape()[1]);
        System.out.println( TAG+" "+" createAndConfigureModel - label.eval(placeholderData).shape()[2] 1-  "+ label.eval(placeholderData).shape()[2]);
        System.out.println( TAG+" "+" createAndConfigureModel - placeholderData.get(\"label\").shape()[0] 1-  "+ placeholderData.get("label").shape()[0]);
        System.out.println( TAG+" "+" createAndConfigureModel - placeholderData.get(\"label\").shape()[1] 1-  "+ placeholderData.get("label").shape()[1]);
        System.out.println( TAG+" "+" createAndConfigureModel - placeholderData.get(\"label\").shape()[2] 1-  "+ placeholderData.get("label").shape()[2]);
        System.out.println( TAG+" "+" createAndConfigureModel - label.eval(placeholderData)) 1-  "+ label.eval(placeholderData));

//        INDArray ouArrayPermuted = outArray.permute(1, 0, 2);

//        INDArray outReducedArray = outArray.get(NDArrayIndex.interval(0, label.getShape()[0]), NDArrayIndex.interval(0, label.getShape()[1]), NDArrayIndex.interval(0, label.getShape()[2]));
        INDArray outReducedArray = outArray.get(NDArrayIndex.interval(0, label.eval(placeholderData).shape()[0]), NDArrayIndex.interval(0, label.eval(placeholderData).shape()[1]), NDArrayIndex.interval(0, label.eval(placeholderData).shape()[2]));
//        INDArray outReducedArray = ouArrayPermuted.get(NDArrayIndex.interval(0, label.eval(placeholderData).shape()[0]), NDArrayIndex.interval(0, label.eval(placeholderData).shape()[1]), NDArrayIndex.interval(0, label.eval(placeholderData).shape()[2]));
//        INDArray outReducedArray = ouArrayPermuted.get(NDArrayIndex.interval(0, trainData.next().getLabels().shape()[0]), NDArrayIndex.interval(0, trainData.next().getLabels().shape()[1]), NDArrayIndex.interval(0, trainData.next().getLabels().shape()[2]));
//        INDArray outReducedArray = ouArrayPermuted.get(NDArrayIndex.interval(0, trainData.next().getLabels().shape()[0]), NDArrayIndex.interval(0, trainData.next().getLabels().shape()[1]), NDArrayIndex.interval(0, trainData.next().getLabels().shape()[2]));
//        INDArray outReducedArray = outArray.get(NDArrayIndex.all(), NDArrayIndex.interval(0, trainData.next().getLabels().shape()[1]), NDArrayIndex.interval(0, trainData.next().getLabels().shape()[2]));
//        INDArray outReducedArray = outArray.get(NDArrayIndex.all(), NDArrayIndex.interval(0, labelArray.shape()[1]), NDArrayIndex.interval(0, labelArray.shape()[2]));
        System.out.println( TAG+" "+" createAndConfigureModel - outReducedArray.shape().length 1-  "+ outReducedArray.shape().length);
        System.out.println( TAG+" "+" createAndConfigureModel - outReducedArray.shape()[0] 1-  "+ outReducedArray.shape()[0]);
        System.out.println( TAG+" "+" createAndConfigureModel - outReducedArray.shape()[1] 1-  "+ outReducedArray.shape()[1]);
        System.out.println( TAG+" "+" createAndConfigureModel - outReducedArray.shape()[2] 1-  "+ outReducedArray.shape()[2]);
        System.out.println( TAG+" "+" createAndConfigureModel - outReducedArray 1-  "+ outReducedArray);


        System.out.println( TAG+" "+" createAndConfigureModel - ======================================================= - ");
        System.out.println( TAG+" "+" createAndConfigureModel - label.eval(placeholderData) - "+ label.eval(placeholderData));
        System.out.println( TAG+" "+" createAndConfigureModel - label.getArr() - "+ label.getArr());

        outReduced = sd.var("outReduced"+" - "+mRandomNumericalId, outReducedArray);
//        outReduced = sd.var("outReduced"+" - "+mRandomNumericalId, outReducedArray);
        System.out.println( TAG+" "+" createAndConfigureModel - outReduced.name() 0-  "+ outReduced.name());
        System.out.println( TAG+" "+" createAndConfigureModel - outReduced.getArr() -  "+ outReduced.getArr());

        INDArray labelArray = label.eval(placeholderData);
//        INDArray labelArray = label.getArr();
        INDArray labelArrayResized = Nd4j.create(outReduced.eval().shape()[0], outReduced.eval().shape()[1], outReduced.eval().shape()[2]);
        INDArray labelArrayResizedPopulated = labelArrayResized.assign(labelArray);
        labelResizedPopulated = sd.var("labelResizedPopulated"+" - "+mRandomNumericalId, labelArrayResizedPopulated);

//        if(!outReducedName.equalsIgnoreCase(""))
//        {
//            outReduced.setVarName(outReducedName);
//            outReducedName = "";
//            System.out.println( TAG+" "+" createAndConfigureModel - outReducedArray.shape()[0] 2-  "+ outReducedArray.shape()[0]);
//            System.out.println( TAG+" "+" createAndConfigureModel - outReducedArray.shape()[1] 2-  "+ outReducedArray.shape()[1]);
//            System.out.println( TAG+" "+" createAndConfigureModel - outReducedArray.shape()[2] 2-  "+ outReducedArray.shape()[2]);
//        }

//        System.out.println( TAG+" "+" createAndConfigureModel - ======================================================= - ");
//        System.out.println( TAG+" "+" createAndConfigureModel - Arrays.toString(labelResizedPopulated.eval().shape()) - "+ Arrays.toString(labelResizedPopulated.eval(placeholderData).shape()));

//        System.out.println( TAG+" "+" createAndConfigureModel - Printing sd information 0--- ");
////            System.out.println(sd.toString());
//        System.out.println(sd.summary());

//        System.out.println( TAG+" "+" createAndConfigureModel - ns.toString() - "+ ns.toString());
//        System.out.println( TAG+" "+" createAndConfigureModel - ns.getName() - "+ ns.getName());

//        INDArray labelResizedArray;
//        SDVariable labelResized;
//        INDArray outReducedResizedArray;
//        SDVariable outReducedResized;
//
//        System.out.println( TAG+" "+" createAndConfigureModel - placeholderData.get(\"label\").shape()[0] 2- "+ placeholderData.get("label").shape()[0]);
//        System.out.println( TAG+" "+" createAndConfigureModel - placeholderData.get(\"label\").shape()[1] 2- "+ placeholderData.get("label").shape()[1]);
//        System.out.println( TAG+" "+" createAndConfigureModel - placeholderData.get(\"label\").shape()[2] 2- "+ placeholderData.get("label").shape()[2]);
//
//        placeholderData.put("input",  trainData.next().getFeatures());
//        placeholderData.put("label", trainData.next().getLabels());
////        trainData.reset();
//
////        USED FOR EXAMPLE OF RESIZING INDARRAY - INDArray outReducedArray = outArray.get(NDArrayIndex.all(), NDArrayIndex.interval(0, dim1), NDArrayIndex.interval(0, dim2));
//
//        System.out.println( TAG+" "+" createAndConfigureModel - placeholderData.get(\"label\").shape()[0] 3- "+ placeholderData.get("label").shape()[0]);
//        System.out.println( TAG+" "+" createAndConfigureModel - placeholderData.get(\"label\").shape()[1] 3- "+ placeholderData.get("label").shape()[1]);
//        System.out.println( TAG+" "+" createAndConfigureModel - placeholderData.get(\"label\").shape()[2] 3- "+ placeholderData.get("label").shape()[2]);
//        System.out.println( TAG+" "+" createAndConfigureModel - outReduced.eval().shape()[0] 2- "+ outReduced.eval().shape()[0]);
//        System.out.println( TAG+" "+" createAndConfigureModel - outReduced.eval().shape()[1] 2- "+ outReduced.eval().shape()[1]);
//        System.out.println( TAG+" "+" createAndConfigureModel - outReduced.eval().shape()[2] 2- "+ outReduced.eval().shape()[2]);
//
//
//        if(placeholderData.get("label").shape()[2] > outReduced.eval().shape()[2])
////            if(tNext.getLabels().size(2) > outReduced.eval().shape()[2])
//        {
//            System.out.println( TAG+" "+" createAndConfigureModel -----------(placeholderData.get(\"label\").shape()[2] > outReduced.eval().shape()[2])------------");
//            labelResizedArray = placeholderData.get("label").get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(0, outReduced.eval().shape()[2]));
//            labelResized = sd.var(labelResizedArray);
//            lossMSE = sd.loss.meanSquaredError("lossMSE", labelResized, outReduced, null);
//        }
//        else if(placeholderData.get("label").shape()[2] < outReduced.eval().shape()[2])
////            if(tNext.getLabels().size(2) <>> outReduced.eval().shape()[2])
//        {
//            System.out.println( TAG+" "+" createAndConfigureModel -----------(placeholderData.get(\"label\").shape()[2] < outReduced.eval().shape()[2])------------");
//            outReducedResizedArray = outReduced.getArr().get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(0, placeholderData.get("label").shape()[2]));
//            outReducedResized = sd.var(outReducedResizedArray);
//            lossMSE = sd.loss.meanSquaredError("lossMSE", label, outReducedResized, null);
//        }
//        else
//        {
//            System.out.println( TAG+" "+" createAndConfigureModel -----------(placeholderData.get(\"label\").shape()[2] = outReduced.eval().shape()[2])------------");
//            lossMSE = sd.loss.meanSquaredError("lossMSE", label, outReduced, null);
//        }
        System.out.println( TAG+" "+" createAndConfigureModel - label.eval(placeholderData).shape()[0] 2-  "+ label.eval(placeholderData).shape()[0]);
        System.out.println( TAG+" "+" createAndConfigureModel - label.eval(placeholderData).shape()[1] 2-  "+ label.eval(placeholderData).shape()[1]);
        System.out.println( TAG+" "+" createAndConfigureModel - label.eval(placeholderData).shape()[2] 2-  "+ label.eval(placeholderData).shape()[2]);
        System.out.println( TAG+" "+" createAndConfigureModel - placeholderData.get(\"label\").shape()[0] 2-  "+ placeholderData.get("label").shape()[0]);
        System.out.println( TAG+" "+" createAndConfigureModel - placeholderData.get(\"label\").shape()[1] 2-  "+ placeholderData.get("label").shape()[1]);
        System.out.println( TAG+" "+" createAndConfigureModel - placeholderData.get(\"label\").shape()[2] 2-  "+ placeholderData.get("label").shape()[2]);
        System.out.println( TAG+" "+" createAndConfigureModel - labelResizedPopulated.name() -  "+ labelResizedPopulated.name());
        System.out.println( TAG+" "+" createAndConfigureModel - labelResizedPopulated.getArr() -  "+ labelResizedPopulated.getArr());
        System.out.println( TAG+" "+" createAndConfigureModel - outReducedArray.shape()[0] 3-  "+ outReducedArray.shape()[0]);
        System.out.println( TAG+" "+" createAndConfigureModel - outReducedArray.shape()[1] 3-  "+ outReducedArray.shape()[1]);
        System.out.println( TAG+" "+" createAndConfigureModel - outReducedArray.shape()[2] 3-  "+ outReducedArray.shape()[2]);

        System.out.println( TAG+" "+" createAndConfigureModel - outReduced.name() 11-  "+ outReduced.name());
        lossMSE = sd.loss.meanSquaredError("lossMSE", labelResizedPopulated, outReduced, null);
//        lossMSE = sd.loss.meanSquaredError("lossMSE", label, outReduced, null);
//        SDVariable lossMSE = sd.loss.meanSquaredError("lossMSE", labelResizedPopulated, out, null);
//        SDVariable lossMSE = sd.loss.meanSquaredError("lossMSE", label, out, null);
        System.out.println( TAG+" "+" createAndConfigureModel - Arrays.toString(lossMSE.getShape()) - "+ Arrays.toString(lossMSE.getShape()));
//        System.out.println( TAG+" "+" createAndConfigureModel - lossMSE.eval(placeholderData).shapeInfoToString() - "+ lossMSE.eval(placeholderData).shapeInfoToString());
//        System.out.println( TAG+" "+" createAndConfigureModel - lossMSE.eval() - "+ lossMSE.eval());
        System.out.println( TAG+" "+" createAndConfigureModel - lossMSE.eval(placeholderData) - "+ lossMSE.eval(placeholderData));
//        System.out.println( TAG+" "+" createAndConfigureModel - lossMSE.eval().shapeInfoToString() - "+ lossMSE.eval().shapeInfoToString());
//        System.out.println( TAG+" "+" createAndConfigureModel - lossMSE.eval() - "+ lossMSE.eval());
        System.out.println( TAG+" "+" createAndConfigureModel - lossMSE.name() - "+ lossMSE.name());
        System.out.println( TAG+" "+" createAndConfigureModel - lossMSE.getArr() -  "+ lossMSE.getArr());

//        SDVariable lossLOG = sd.loss.logLoss("lossLOG", label, out);
////            SDVariable loss = sd.loss.logLoss("loss", labelPadded, out);

        System.out.println( TAG+" "+" createAndConfigureModel - ======================================================= - ");
//        System.out.println( TAG+" "+" createAndConfigureModel - Arrays.toString(lossMSE.eval(placeholderData).shape()) - "+ Arrays.toString(lossMSE.eval(placeholderData).shape()));
//        System.out.println( TAG+" "+" ======================================================= - ");
//        System.out.println( TAG+" "+" Arrays.toString(lossLOG.eval().shape()) - "+ Arrays.toString(lossLOG.eval(placeholderData).shape()));

        sd.setLossVariables(lossMSE);
//        sd.setLossVariables(lossMSE.name());
//        sd.setLossVariables("lossMSE");
//        sd.setLossVariables(ns.getName()+"/"+"lossMSE");
//        sd.setLossVariables("lossLOG");


//    number of iterations for LRF
        int iterations = 70;

//   creating configuration
//        optimizer = torch.optim.SGD(tf_model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-3, nesterov=True)
//    # optimizer = torch.optim.Adam(tf_model.parameters(), lr=1e-4)

        System.out.println( TAG+" "+" createAndConfigureModel - outReduced.name() 1-  "+ outReduced.name());

        evaluation = new Evaluation();
//        Evaluation evaluation = new Evaluation();
        double learningRate = 1e-3;
        TrainingConfig config = new TrainingConfig.Builder()
//                .l2(1e-4)                               //L2 regularization
//                .l2(0.001)
//                .updater(new Adam(learningRate))        //Adam optimizer with specified learning rate
                .updater(new Nesterovs(0.01, 0.9))
//                .updater(new Nesterovs(0.0001, 0.9))
                .dataSetFeatureMapping("input")         //DataSet features array should be associated with variable "input"
                .dataSetLabelMapping("label")           //DataSet label array should be associated with variable "label"
//                .minimize("lossLOG")
//                .minimize(lossMSE.name())
//                .minimize("lossMSE")
//                .trainEvaluation(outReduced.name(), 0, evaluation)  // add a training evaluation
                .trainEvaluation(outReduced.name(), 0, evaluation)  // add a training evaluation
//                .trainEvaluation("outReduced"+" - "+mRandomNumericalId, 0, evaluation)  // add a training evaluation
//                .trainEvaluation("out", 0, evaluation)  // add a training evaluation
                .build();

        sd.setTrainingConfig(config);

        System.out.println( TAG+" "+" createAndConfigureModel - Printing sd information 1--- ");
//            System.out.println(sd.toString());
        System.out.println(sd.summary());

    }

    private static void prepareTrainingAndTestData()
    {

        System.out.println( TAG+" "+" prepareTrainingAndTestData - lastTrainCount - 0 - "+lastTrainCount);
        System.out.println( TAG+" "+" prepareTrainingAndTestData - lastTestCount - 0 - "+lastTestCount);

        batch_size = 128;
//        batch_size = 256;
//        batch_size = emb_size / ((int) (emb_size / (lastTestCount)));
//        batch_size = emb_size / ((int) (emb_size / (lastTestCount/2)) - 1);
//        batch_size = lastTestCount/2;

        miniBatchSize = 128;
//        miniBatchSize = 256;
//        miniBatchSize = emb_size / ((int) (emb_size / (lastTestCount)));
//        miniBatchSize = emb_size / ((int) (emb_size / (lastTestCount/2)) - 1);
//        miniBatchSize = lastTestCount/2;

        System.out.println( TAG+" "+" prepareTrainingAndTestData - batch_size - 0 - "+batch_size);
        System.out.println( TAG+" "+" prepareTrainingAndTestData - miniBatchSize - 0 - "+miniBatchSize);

        System.out.println( TAG+" "+" prepareTrainingAndTestData - (lastTrainCount - lastTrainCount%miniBatchSize - 1) - 0 - "+(lastTrainCount - lastTrainCount%miniBatchSize - 1));
        System.out.println( TAG+" "+" prepareTrainingAndTestData - (lastTestCount - lastTestCount%miniBatchSize - 1) - 0 - "+(lastTestCount - lastTestCount%miniBatchSize - 1));

        // ----- Load the training data -----
        trainFeatures = new CSVSequenceRecordReader();
//        trainFeatures.initialize(new NumberedFileInputSplit(featuresDirTrain.getAbsolutePath() + "/%d.csv", 0, lastTrainCount));
        try {
            trainFeatures.initialize(new NumberedFileInputSplit(featuresDirTrain.getAbsolutePath() + "/%d.csv", 0, lastTrainCount - lastTrainCount%miniBatchSize - 1));
        } catch (IOException e) {
            throw new RuntimeException(e);
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
        trainLabels = new CSVSequenceRecordReader();
//        trainLabels.initialize(new NumberedFileInputSplit(labelsDirTrain.getAbsolutePath() + "/%d.csv", 0, lastTrainCount));
        try {
            trainLabels.initialize(new NumberedFileInputSplit(labelsDirTrain.getAbsolutePath() + "/%d.csv", 0, lastTrainCount - lastTrainCount%miniBatchSize - 1));
        } catch (IOException e) {
            throw new RuntimeException(e);
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }

//        System.out.println( TAG+" "+" - trainFeatures.next().toString() - " + trainFeatures.next().toString());
//        System.out.println( TAG+" "+" - trainLabels.next().toString() - " + trainLabels.next().toString());

        trainData = new SequenceRecordReaderDataSetIterator(trainFeatures, trainLabels, miniBatchSize, numLabelClasses,
                true, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);


        // ----- Load the test data -----
        //Same process as for the training data.
        System.out.println( TAG+" "+" prepareTrainingAndTestData - lastTestCount%miniBatchSize - "+lastTestCount%miniBatchSize);
        if(lastTestCount > lastTestCount%miniBatchSize + 1)
        {
            testFeatures = new CSVSequenceRecordReader();
            try {
//        testFeatures.initialize(new NumberedFileInputSplit(featuresDirTest.getAbsolutePath() + "/%d.csv", 0, lastTestCount));
                testFeatures.initialize(new NumberedFileInputSplit(featuresDirTest.getAbsolutePath() + "/%d.csv", 0, lastTestCount - lastTestCount%miniBatchSize - 1));
            } catch (IOException e) {
                throw new RuntimeException(e);
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
            testLabels = new CSVSequenceRecordReader();
            try {
//        testLabels.initialize(new NumberedFileInputSplit(labelsDirTest.getAbsolutePath() + "/%d.csv", 0, lastTestCount));
                testLabels.initialize(new NumberedFileInputSplit(labelsDirTest.getAbsolutePath() + "/%d.csv", 0, lastTestCount - lastTestCount%miniBatchSize - 1));
            } catch (IOException e) {
                throw new RuntimeException(e);
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }
        else
        {
            testFeatures = new CSVSequenceRecordReader();
            try {
        testFeatures.initialize(new NumberedFileInputSplit(featuresDirTest.getAbsolutePath() + "/%d.csv", 0, lastTestCount));
//                testFeatures.initialize(new NumberedFileInputSplit(featuresDirTest.getAbsolutePath() + "/%d.csv", 0, lastTestCount - lastTestCount%miniBatchSize - 1));
            } catch (IOException e) {
                throw new RuntimeException(e);
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
            testLabels = new CSVSequenceRecordReader();
            try {
        testLabels.initialize(new NumberedFileInputSplit(labelsDirTest.getAbsolutePath() + "/%d.csv", 0, lastTestCount));
//                testLabels.initialize(new NumberedFileInputSplit(labelsDirTest.getAbsolutePath() + "/%d.csv", 0, lastTestCount - lastTestCount%miniBatchSize - 1));
            } catch (IOException e) {
                throw new RuntimeException(e);
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }

        }

        testData = new SequenceRecordReaderDataSetIterator(testFeatures, testLabels, miniBatchSize, numLabelClasses,
                true, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        trainData.reset();
        testData.reset();
        System.out.println( TAG+" "+" prepareTrainingAndTestData - Printing traindata dataset shape - 0 - ");
        while(trainData.hasNext()) {
            DataSet data = trainData.next();
            System.out.println(Arrays.toString(data.getFeatures().shape()));
        }
        System.out.println( TAG+" "+" prepareTrainingAndTestData - Printing testdata dataset shape - 0 - ");
        while(testData.hasNext()) {
            DataSet data2 = testData.next();
            System.out.println(Arrays.toString(data2.getFeatures().shape()));
        }
        trainData.reset();
        testData.reset();

        System.out.println( TAG+" "+" prepareTrainingAndTestData - Printing traindata dataset shape - 1 - ");
        DataSet data = trainData.next();
        System.out.println(Arrays.toString(data.getFeatures().shape()));

        System.out.println( TAG+" "+" prepareTrainingAndTestData - Printing testdata dataset shape - 1 - ");
        DataSet data2 = testData.next();
        System.out.println(Arrays.toString(data2.getFeatures().shape()));

        normalizer = new NormalizerStandardize();
        normalizer.fitLabel(true);
        normalizer.fit(trainData);           //Collect the statistics (mean/stdev) from the training data. This does not modify the input data

        trainData.reset();
        testData.reset();

//        int index = 0;
        while(trainData.hasNext()) {
//            ++index;
//            System.out.println( TAG+" "+" index - " + index);
            normalizer.transform(trainData.next());     //Apply normalization to the training data
        }

//        index = 0;
        while(testData.hasNext()) {
//            ++index;
//            System.out.println( TAG+" "+" index - " + index);
            normalizer.transform(testData.next());         //Apply normalization to the test data. This is using statistics calculated from the *training* set
        }

        trainData.reset();
        testData.reset();

        trainData.setPreProcessor(normalizer);
        testData.setPreProcessor(normalizer);

        System.out.println( TAG+" "+" prepareTrainingAndTestData - Printing traindata dataset shape - 2");
        while(trainData.hasNext()) {
            data = trainData.next();
            System.out.println(Arrays.toString(data.getFeatures().shape()));
        }
        System.out.println( TAG+" "+" prepareTrainingAndTestData - Printing testdata dataset shape - 2");
        while(testData.hasNext()) {
            data2 = testData.next();
            System.out.println(Arrays.toString(data2.getFeatures().shape()));
        }

        trainData.reset();
        testData.reset();

    }


    public static void Ui(File testDir) throws Exception {
        testDir = baseAssetDir;
        File dir = testDir;
        File f = new File(dir, "ui_data.bin");
        System.out.println( TAG+" "+" Ui - File path: {} - "+f.getAbsolutePath());

        f.getParentFile().mkdirs();
        f.delete();

        LogFileWriter lfw = new LogFileWriter(f);
        lfw.writeGraphStructure(sd);
        lfw.writeFinishStaticMarker();

        //Append a number of events
        lfw.registerEventName("accuracy");
        lfw.registerEventName("precision");
        long t = System.currentTimeMillis();
        int numberOfIterations = 2;
        for (int iter = 0; iter < numberOfIterations; iter++) {
            double d = Math.cos(0.1 * iter);
            d *= d;
            lfw.writeScalarEvent("accuracy", LogFileWriter.EventSubtype.EVALUATION, t + iter, iter, 0, d);

            double prec = Math.min(0.05 * iter, 1.0);
            lfw.writeScalarEvent("precision", LogFileWriter.EventSubtype.EVALUATION, t + iter, iter, 0, prec);
        }

        //Add some histograms:
        lfw.registerEventName("histogramDiscrete");
        lfw.registerEventName("histogramEqualSpacing");
        lfw.registerEventName("histogramCustomBins");
        for (int i = 0; i < 3; i++) {
            INDArray discreteY = Nd4j.createFromArray(0, 1, 2);
            lfw.writeHistogramEventDiscrete("histogramDiscrete", LogFileWriter.EventSubtype.TUNING_METRIC, t + i, i, 0, Arrays.asList("zero", "one", "two"), discreteY);

            INDArray eqSpacingY = Nd4j.createFromArray(-0.5 + 0.5 * i, 0.75 * i + i, 1.0 * i + 1.0);
            lfw.writeHistogramEventEqualSpacing("histogramEqualSpacing", LogFileWriter.EventSubtype.TUNING_METRIC, t + i, i, 0, 0.0, 1.0, eqSpacingY);

            INDArray customBins = Nd4j.createFromArray(new double[][]{
                    {0.0, 0.5, 0.9},
                    {0.2, 0.55, 1.0}
            });
            System.out.println(Arrays.toString(customBins.data().asFloat()));
            System.out.println(customBins.shapeInfoToString());
            lfw.writeHistogramEventCustomBins("histogramCustomBins", LogFileWriter.EventSubtype.TUNING_METRIC, t + i, i, 0, customBins, eqSpacingY);
        }

//        uiServer = UIServer.getInstance();

    }

        private static void getConfiguration()
    {

        placeholderData = new HashMap<>();

        placeholderData.put("input",  t.getFeatures());
        placeholderData.put("label", t.getLabels());

//        SDVariable inputDataSet0 = sd.var("inputDataSet0", t.getFeatures());
//        SDVariable labelDataSet0 = sd.var("labelDataSet0", t.getLabels());

        NDBase mNDBase = new NDBase();
        INDArray inputDataSetDim0Size = mNDBase.sizeAt(t.getFeatures(), 0);
        INDArray labelDataSetDim0Size = mNDBase.sizeAt(t.getLabels(), 0);

//        System.out.println( TAG+" "+" features - t.getFeatures().size(0) - 0 - "+t.getFeatures().size(0));
//        System.out.println( TAG+" "+" features - t.getFeatures().size(1) - 0 - "+t.getFeatures().size(1));
//        System.out.println( TAG+" "+" features - t.getFeatures().size(2) - 0 - "+t.getFeatures().size(2));
//        System.out.println( TAG+" "+" labels - t.getLabels().size(0) - 0 - "+t.getLabels().size(0));
//        System.out.println( TAG+" "+" labels - t.getLabels().size(1) - 0 - "+t.getLabels().size(1));
//        System.out.println( TAG+" "+" labels - t.getLabels().size(2) - 0 - "+t.getLabels().size(2));
//
//        System.out.println( TAG+" "+" features - inputDataSetDim0Size - "+inputDataSetDim0Size);
//        System.out.println( TAG+" "+" labels - labelDataSetDim0Size - "+labelDataSetDim0Size);

        //Create input and label variables
//        SDVariable in = sd.placeHolder("input", DataType.FLOAT, -1);
//        SDVariable label = sd.placeHolder("label", DataType.FLOAT, -1);
//        SDVariable in = sd.placeHolder("input", DataType.FLOAT, 32, nIn);
//        SDVariable label = sd.placeHolder("label", DataType.FLOAT, 32, nOut);
//        SDVariable input = sd.placeHolder("input", DataType.FLOAT, miniBatchSize, nIn, -1);
//        SDVariable label = sd.placeHolder("label", DataType.FLOAT, miniBatchSize, nOut, 1);
//            SDVariable input = sd.placeHolder("input", DataType.FLOAT, miniBatchSize, nIn, -1);
//            SDVariable label = sd.placeHolder("label", DataType.FLOAT, miniBatchSize, nOut, 1);
//        SDVariable input = sd.placeHolder("input", DataType.FLOAT, miniBatchSize, nIn, t.getFeatures().size(2));
//        SDVariable label = sd.placeHolder("label", DataType.FLOAT, miniBatchSize, nOut, t.getLabels().size(2));
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, nIn, -1);
        SDVariable label = sd.placeHolder("label", DataType.FLOAT, -1, nOut, -1);
//        SDVariable input = sd.placeHolder("input", DataType.FLOAT, inputDataSetDim0Size.getInt(0), nIn, -1);
//        SDVariable label = sd.placeHolder("label", DataType.FLOAT, labelDataSetDim0Size.getInt(0), nOut, -1);
//        SDVariable input = sd.placeHolder("input", DataType.FLOAT, miniBatchSize, nIn, -1);
//        SDVariable label = sd.placeHolder("label", DataType.FLOAT, miniBatchSize, nOut, -1);
//            SDVariable label = sd.placeHolder("label", DataType.FLOAT, miniBatchSize, 4);

//        System.out.println( TAG+" "+" input.toString() - "+input.toString());

        //Define LSTM layer

        LSTMLayerConfig mLSTMConfiguration = LSTMLayerConfig.builder()
                .lstmdataformat(LSTMDataFormat.NST)
//                .lstmdataformat(LSTMDataFormat.NTS)
                .directionMode(LSTMDirectionMode.FWD)
//                .directionMode(LSTMDirectionMode.BIDIR_CONCAT)
                .gateAct(LSTMActivations.SIGMOID)
                .cellAct(LSTMActivations.TANH)
                .outAct(LSTMActivations.TANH)
//                .cellAct(LSTMActivations.SOFTPLUS)
//                .outAct(LSTMActivations.SOFTPLUS)
                .retFullSequence(true)
                .retLastC(false)
                .retLastH(true)
                .build();

//        System.out.println( TAG+" "+" mLSTMConfiguration.toString() - "+mLSTMConfiguration.toString());

//IMPLEMENTATION - TWO - START - (--)(--)(--)(--)(--)(--)(--)(--)(--)(--)(--)(--)(--)(--)(--)(--)(--)(--)(--)(--)(--)(--)(--)(--)(--)(--)(--)(--)

//            System.out.println( TAG+" "+" ======================================================= - ");
//            System.out.println( TAG+" "+" input.toString() - 1 - "+input.toString());
//            System.out.println( TAG+" "+" whileLoopIndex - "+whileLoopIndex);
//            System.out.println( TAG+" "+" dim2 - "+dim2);

        LSTMLayerOutputs outputs = new LSTMLayerOutputs(sd.rnn.lstmLayer(
                input,
//                input, cLast, yLast, null,
                LSTMLayerWeights.builder()
//                        .weights(W)
//                        .rWeights(rW)
//                        .iWeights(iW.getArr())
//                        .bias(b)
//                            .weights(sd.var("weights", Nd4j.rand(DataType.FLOAT, 2, nOut, 4 * nOut)))
                        .weights(sd.var("weights", Nd4j.rand(DataType.FLOAT, nIn, 4 * nOut)))
//                        .weights(sd.var("weights", Nd4j.rand(DataType.FLOAT, 2, t.getFeatures().size(2), 4 * nOut)))
//                        .weights(sd.var("weights", Nd4j.rand(DataType.FLOAT, 2, nIn, 4 * nOut)))
                        .rWeights(sd.var("rWeights", Nd4j.rand(DataType.FLOAT, nOut, 4 * nOut)))
//                            .rWeights(sd.var("rWeights", Nd4j.rand(DataType.FLOAT, nIn, 4 * nOut)))
//                            .rWeights(sd.var("rWeights", Nd4j.rand(DataType.FLOAT, nOut, 4 * numUnits)))
//                            .rWeights(sd.var("rWeights", Nd4j.rand(DataType.FLOAT, t.getFeatures().size(2), 4 * nOut)))
//                        .peepholeWeights(sd.var("inputPeepholeWeights", Nd4j.rand(DataType.FLOAT, 3 * nOut)))
                        .bias(sd.var("bias", Nd4j.rand(DataType.FLOAT, 4 * nOut)))
//                        .bias(sd.var("bias", Nd4j.rand(DataType.FLOAT, 2, 4 * nOut)))
                        .build(),
                mLSTMConfiguration), mLSTMConfiguration);

        // t.getFeatures().size(0) == input.getShape()[0] == miniBatchSize
        // t.getFeatures().size(1) == input.getShape()[1] == nIn
        // t.getFeatures().size(2) == input.getShape()[2] == TimeSteps

        System.out.println( TAG+" "+" features - t.getFeatures().size(0) - 0 - "+t.getFeatures().size(0));
        System.out.println( TAG+" "+" features - t.getFeatures().size(1) - 0 - "+t.getFeatures().size(1));
        System.out.println( TAG+" "+" features - t.getFeatures().size(2) - 0 - "+t.getFeatures().size(2));

        System.out.println( TAG+" "+" input.getShape()[0] - "+input.getShape()[0]);
        System.out.println( TAG+" "+" input.getShape()[1] - "+input.getShape()[1]);
        System.out.println( TAG+" "+" input.getShape()[2] - "+input.getShape()[2]);

//            System.out.println( TAG+" "+" ======================================================= - ");
//            System.out.println( TAG+" "+" trainData.next(whileLoopIndex+1).getFeatures().size(2) - "+trainData.next(whileLoopIndex+1).getFeatures().size(2));
//            System.out.println( TAG+" "+" ======================================================= - ");
//            System.out.println( TAG+" "+" dim2 - "+dim2);

//            int trainDataCopySize = Iterators.size(trainDataCopy);
//            System.out.println( TAG+" "+" ======================================================= - ");
//            System.out.println( TAG+" "+" trainDataCopySize - "+trainDataCopySize);
//
//            int trainDataCopyIndex = 0;
//
//            trainDataCopy.reset();
//
//            while(trainDataCopy.hasNext())
//            {
//
//                System.out.println( TAG+" "+" Processing trainDataCopy for updated dim2 - ");
//
//                DataSet trainDataCopyDataSet = trainDataCopy.next();
//
//                if(trainDataCopyDataSet.getFeatures().size(2) != dim2)
//                {
//                    System.out.println( TAG+" "+" ======================================================= - ");
//                    System.out.println( TAG+" "+" trainDataCopyIndex - "+trainDataCopyIndex);
//                    System.out.println( TAG+" "+" trainDataCopyDataSet.getFeatures().size(2) - "+trainDataCopyDataSet.getFeatures().size(2));
//                    System.out.println( TAG+" "+" ======================================================= - ");
//                    System.out.println( TAG+" "+" dim2 - "+dim2);
//                    break;
//                }
//
//            }

//            System.out.println( TAG+" "+" ======================================================= - ");
//            System.out.println( TAG+" "+" trainData.next(whileLoopIndex+43).getFeatures().size(2) - "+trainData.next(whileLoopIndex+43).getFeatures().size(2));
//            System.out.println( TAG+" "+" ======================================================= - ");
//            System.out.println( TAG+" "+" dim2 - "+dim2);
//            System.out.println( TAG+" "+" ======================================================= - ");
//            System.out.println( TAG+" "+" outputs.toString() - 1 - "+outputs.toString());
//            System.out.println( TAG+" "+" ======================================================= - ");
//            System.out.println( TAG+" "+" whileLoopIndex - "+whileLoopIndex);

//           Behaviour with default settings: 3d (time series) input with shape
//          [miniBatchSize, vectorSize, timeSeriesLength] -> 2d output [miniBatchSize, vectorSize]

        SDVariable layer0 = outputs.getOutput();

        System.out.println( TAG+" "+" ======================================================= - ");
        System.out.println( TAG+" "+" t.getFeatures().size(2) - "+ t.getFeatures().size(2));
        System.out.println( TAG+" "+" miniBatchSize) - "+ miniBatchSize);
        System.out.println( TAG+" "+" nOut - "+ nOut);

         System.out.println( TAG+" "+" ======================================================= - ");
         System.out.println( TAG+" "+" Arrays.toString(layer0.eval().shape()) - 0 - "+ Arrays.toString(layer0.eval(placeholderData).shape()));

        SDVariable layer0Permuted = sd.permute(layer0, 0, 2, 1);

        System.out.println( TAG+" "+" ======================================================= - ");
        System.out.println( TAG+" "+" Arrays.toString(layer0Permuted.eval().shape()) - "+ Arrays.toString(layer0Permuted.eval(placeholderData).shape()));

        SDVariable layer0PermutedReshaped = sd.reshape(layer0Permuted, -1, nOut);
//        SDVariable layer0PermutedReshaped = sd.reshape(layer0Permuted, miniBatchSize * (int)t.getFeatures().size(2), nOut);

        System.out.println( TAG+" "+" ======================================================= - ");
        System.out.println( TAG+" "+" Arrays.toString(layer0PermutedReshaped.eval().shape()) - "+ Arrays.toString(layer0PermutedReshaped.eval(placeholderData).shape()));

        //            SDVariable layer1 = layer0.mean(1);

//        SDVariable weightsPlaceholder = sd.placeHolder("weightsPlaceholder", DataType.FLOAT, miniBatchSize, t.getFeatures().size(2), t.getFeatures().size(2));
//        INDArray weightsPlaceholderValue = Nd4j.rand(DataType.FLOAT, miniBatchSize, t.getFeatures().size(2), t.getFeatures().size(2));
//        SDVariable weightsPlaceholder = sd.var("weightsPlaceholder", weightsPlaceholderValue);

////        SDVariable w1 = sd.var("w1", Nd4j.rand(DataType.FLOAT, 2, nOut, 2));
////        SDVariable b1 = sd.var("b1", Nd4j.rand(DataType.FLOAT, 2, nOut));
////        SDVariable w1 = sd.var("w1", Nd4j.rand(DataType.FLOAT, 2, 57, nOut));
////        SDVariable b1 = sd.var("b1", Nd4j.rand(DataType.FLOAT, 2, nOut));
////            SDVariable w1 = sd.var("w1", Nd4j.rand(DataType.FLOAT, nOut, 1));   //1 is the value of numLabelClasses
////            SDVariable w1 = sd.var("w1", Nd4j.rand(DataType.FLOAT, 2, layer0.getArr().size(2), layer0.getArr().size(2)));
////            SDVariable w1 = sd.var("w1", new XavierInitScheme('c', nIn, nOut), DataType.FLOAT, 2, dim2, dim2);
////            SDVariable w1 = sd.var("w1", new XavierInitScheme('c', nIn, nOut), DataType.FLOAT, dim2, dim2);
////            SDVariable w1 = sd.var("w1", new XavierInitScheme('c', nIn, nOut), DataType.FLOAT, 2, nOut, dim2);
////            SDVariable w1 = sd.var("w1", new XavierInitScheme('c', nIn, nOut), DataType.FLOAT, 2, nIn, dim2);
////            SDVariable w1 = sd.var("w1", new XavierInitScheme('c', nIn, nOut), DataType.FLOAT, miniBatchSize, nIn, 4);
////            SDVariable w1 = sd.var("w1", new XavierInitScheme('c', nIn, nOut), DataType.FLOAT,  nIn, 4);
////            SDVariable w1 = sd.var("w1", new XavierInitScheme('c', nIn, nOut), DataType.FLOAT,  4, 4, 28);
////            SDVariable w1 = sd.var("w1", new XavierInitScheme('c', nIn, nOut), DataType.FLOAT,  4, 4);
////            SDVariable w1 = sd.var("w1", new XavierInitScheme('c', nIn, nOut), DataType.FLOAT, miniBatchSize, input.getShape()[1], input.getShape()[2]);
////            SDVariable w1 = sd.var("w1", new XavierInitScheme('c', nIn, nOut), DataType.FLOAT, miniBatchSize, 2, input.getShape()[2]);
//        SDVariable w1 = sd.var("w1", new XavierInitScheme('c', nIn, nOut), DataType.FLOAT, miniBatchSize, t.getFeatures().size(2), t.getFeatures().size(2));
//        SDVariable w1 = sd.var("w1", new XavierInitScheme('c', nIn, nOut), DataType.FLOAT, t.getFeatures().size(2), labelCount);
        SDVariable w1 = sd.var("w1", new XavierInitScheme('c', nIn, nOut), DataType.FLOAT, nOut, labelCount);
        SDVariable b1 = sd.constant("b1", 0.05);
//        SDVariable b1 = sd.var("b1", Nd4j.rand(DataType.FLOAT, miniBatchSize, labelCount, t.getFeatures().size(2)));
//        SDVariable b1 = sd.var("b1", Nd4j.rand(DataType.FLOAT, labelCount));

//        SDVariable mmulOutput = layer0PermutedReshaped.mmul(w1);
        SDVariable mmulOutput = layer0PermutedReshaped.mmul(w1).add(b1);
//        SDVariable mmulOutput = layer0.mmul(w1).add(b1);

        System.out.println( TAG+" "+" ======================================================= - ");
        System.out.println( TAG+" "+" Arrays.toString(mmulOutput.eval().shape()) - "+ Arrays.toString(mmulOutput.eval(placeholderData).shape()));

//        SDVariable mmulOutputUnreshaped = sd.reshape(mmulOutput, sd.sizeAt(label, 0).getArr().getInt(0), -1, labelCount);
//        SDVariable mmulOutputUnreshaped = sd.reshape(mmulOutput, mNDBase.sizeAt(trainData.next().getFeatures(), 0).getInt(0), -1, labelCount);
        SDVariable mmulOutputUnreshaped = sd.reshape(mmulOutput, miniBatchSize, -1, labelCount);
//        SDVariable mmulOutputUnreshaped = sd.reshape(mmulOutput, miniBatchSize, (int)t.getFeatures().size(2), labelCount);

        System.out.println( TAG+" "+" ======================================================= - ");
        System.out.println( TAG+" "+" Arrays.toString(mmulOutputUnreshaped.eval().shape()) - "+ Arrays.toString(mmulOutputUnreshaped.eval(placeholderData).shape()));

        SDVariable mmulOutputUnreshapedUnPermuted = sd.permute(mmulOutputUnreshaped, 0, 2, 1);

        System.out.println( TAG+" "+" ======================================================= - ");
        System.out.println( TAG+" "+" Arrays.toString(mmulOutputUnreshapedUnPermuted.eval().shape()) - "+ Arrays.toString(mmulOutputUnreshapedUnPermuted.eval(placeholderData).shape()));

//        SDVariable out = sd.nn.softmax("out", layer0);
        SDVariable out = sd.nn.softmax("out", mmulOutputUnreshapedUnPermuted);
//        SDVariable out = sd.nn.softmax("out", mmulOutputUnreshapedUnPermuted.add(b1));
//        SDVariable out = sd.nn.softmax("out", layer0.mmul(w1).add(b1));
//        SDVariable out = sd.nn.softmax("out", layer0.mmul(w1FromArray).add(b1));
//        SDVariable out = sd.nn.softmax("out", layer0.mmul(weightsPlaceholder).add(b1));
//            SDVariable out = sd.nn.softmax("out", layer0);
//            SDVariable out = sd.nn.softmax("out", layer1.mmul(w1).add(b1));

        System.out.println( TAG+" "+" ======================================================= - ");
        System.out.println( TAG+" "+" Arrays.toString(out.eval().shape()) - "+ Arrays.toString(out.eval(placeholderData).shape()));
        System.out.println( TAG+" "+" ======================================================= - ");
        System.out.println( TAG+" "+" Arrays.toString(label.eval().shape()) - "+ Arrays.toString(label.eval(placeholderData).shape()));

//            SDNN mSDNN = new SDNN(sd);
////            SDVariable padding = sd.var("padding", Nd4j.zeros(DataType.FLOAT, 3,3));
////            SDVariable padding = sd.constant("padding", Nd4j.zeros(DataType.INT16, 4,dim2));
////            SDVariable padding = sd.constant("padding", Nd4j.zeros(DataType.INT16, 3,2));
//            SDVariable padding = sd.constant("padding", Nd4j.zeros(DataType.INT16, 32,4,dim2));
////            System.out.println( TAG+" "+" padding.etArr().toStringFull() - "+padding.getArr().toStringFull());
//            System.out.println( TAG+" "+" padding.etArr().toStringFull() - "+ Arrays.toString(padding.getShape()));
//
//
//            System.out.println( TAG+" "+" label.toString() - 0 - "+label.toString());
////            System.out.println( TAG+" "+" Arrays.toString(label.getArr().shape()) - 0 - "+Arrays.toString(label.getArr().shape()));
//            SDVariable labelPadded = sd.nn.pad("labelPadded", label, padding, PadMode.CONSTANT, 0.0);
////            System.out.println( TAG+" "+" Arrays.toString(label.getArr().shape()) - 1 - "+Arrays.toString(label.getArr().shape()));
////            System.out.println( TAG+" "+" labelPadded.toString() - 1 - "+labelPadded.toString());
//            System.out.println( TAG+" "+" labelPadded.toString() - 1 - "+ Arrays.toString(labelPadded.getShape()));


//IMPLEMENTATION - TWO - END - (--)(--)(--)(--)(--)(--)(--)(--)(--)(--)(--)(--)(--)(--)(--)(--)(--)(--)(--)(--)(--)(--)(--)(--)(--)(--)(--)(--)


//        File dir = testDir.toFile();
        File f = new File(baseAssetDir, "logFile.bin");
        UIListener l = UIListener.builder(f)
                .plotLosses(1)
                .trainEvaluationMetrics("out", 0, Evaluation.Metric.ACCURACY, Evaluation.Metric.F1)
                .updateRatios(1)
                .build();

        sd.setListeners(l);

        Evaluation evaluation = new Evaluation();
        //Create and set the training configuration
        double learningRate = 1e-3;
        TrainingConfig config = new TrainingConfig.Builder()
//                .l2(1e-4)                               //L2 regularization
                .l2(0.001)
//                .updater(new Adam(learningRate))        //Adam optimizer with specified learning rate
                .updater(new Nesterovs(0.001, 0.9))
                .dataSetFeatureMapping("input")         //DataSet features array should be associated with variable "input"
                .dataSetLabelMapping("label")           //DataSet label array should be associated with variable "label"
//                .minimize("lossLOG")
                .minimize("lossMSE")
//                .trainEvaluation(outReduced.name(), 0, evaluation)  // add a training evaluation
//                .trainEvaluation("outReduced", 0, evaluation)  // add a training evaluation
                .trainEvaluation("out", 0, evaluation)  // add a training evaluation
                .build();

        SDVariable lossMSE = sd.loss.meanSquaredError("lossMSE", label, out, null);

//        SDVariable lossLOG = sd.loss.logLoss("lossLOG", label, out);
////            SDVariable loss = sd.loss.logLoss("loss", labelPadded, out);

        System.out.println( TAG+" "+" ======================================================= - ");
        System.out.println( TAG+" "+" Arrays.toString(lossMSE.eval().shape()) - "+ Arrays.toString(lossMSE.eval(placeholderData).shape()));
//        System.out.println( TAG+" "+" ======================================================= - ");
//        System.out.println( TAG+" "+" Arrays.toString(lossLOG.eval().shape()) - "+ Arrays.toString(lossLOG.eval(placeholderData).shape()));

        sd.setLossVariables("lossMSE");
//        sd.setLossVariables("lossLOG");

        sd.setTrainingConfig(config);

        System.out.println( TAG+" "+" Printing sd information");
//            System.out.println(sd.toString());
        System.out.println(sd.summary());

    }

    public static void MeanAndStandardDeviationOfTrainDataSet() {

        // Concatenate 'src' and 'trg' arrays along axis 1
        INDArray concatenated = Nd4j.concat(1,
                trainData.next().getFeatures().get(all(), interval(1, trainData.next().getFeatures().columns()), interval(2, 4)),
                trainData.next().getFeatures().get(all(), all(), interval(2, 4)));

        // Calculate mean and std along axes 0 and 1
        INDArray meanArray = concatenated.mean(0, 1);
        INDArray stdArray = concatenated.std(0, 1);

        // Initialize arrays to store means and stds for different datasets
        List<INDArray> means = new ArrayList<>();
        List<INDArray> stds = new ArrayList<>();

        int index = 1;
while(trainData.hasNext())
{
        double datasetVal = trainData.next().getFeatures().getDouble(index);

            // Create a mask to filter data for the current dataset
//            INDArray mask = trainData.next().getFeatures().eq(datasetVal);

            // Apply the mask and calculate mean and std
//            INDArray datasetData = concatenated.get(mask, all(), all());

            means.add(trainData.next().getFeatures().mean(0, 1));
            stds.add(trainData.next().getFeatures().std(0, 1));
            ++index;
        }

        // Calculate mean and std for all datasets combined
        INDArray combinedMeans = Nd4j.vstack(means).mean(0);
        INDArray combinedStds = Nd4j.vstack(stds).mean(0);

        mean = sd.var("mean", meanArray);
        std = sd.var("std", stdArray);
//        sd.associateArrayWithVariable(combinedMeans, mean);
//        sd.associateArrayWithVariable(combinedStds, std);

//        // Concatenate the source and target datasets along the second dimension
//        SDVariable concatenated = sd.concat(1, (sd.var(trainData.next().getFeatures().get(all())), interval(1, trainData.next().getFeatures().size(1)), interval(2, 4),
//                sd.var(trainData.next().getFeatures().get(all())), all(), interval(2, 4));
//
//
//
//        // Calculate the mean along dimensions 0 and 1
//        mean = concatenated.mean(0, 1);
//        // Calculate the standard deviation along dimensions 0 and 1
//        std = concatenated.std(false, 0, 1);
//
//        // Get the results as ND4J arrays
//        INDArray meanResult = mean.getArr();
//        INDArray stdResult = std.getArr();
//
//        // Print the results
//        System.out.println( TAG+" "+"Mean: ");
//        System.out.println(meanResult);
//
//        System.out.println( TAG+" "+"Standard Deviation: ");
//        System.out.println(stdResult);

    }

    public static void MeanAndStdCalculator()
    {

        mRandom = new Random();
        mRandomNumericalId = mRandom.nextInt(1000000);

        trainData.reset();;
        INDArray trainDataDatasetHolder = trainData.next().getFeatures();
        System.out.println( TAG+" "+" MeanAndStdCalculator - trainDataDatasetHolder.shape().length -  "+ trainDataDatasetHolder.shape().length);
        System.out.println( TAG+" "+" MeanAndStdCalculator - trainDataDatasetHolder.shape().shape()[0] -  "+ trainDataDatasetHolder.shape()[0]);
        System.out.println( TAG+" "+" MeanAndStdCalculator - trainDataDatasetHolder.shape().shape()[1] -  "+ trainDataDatasetHolder.shape()[1]);
        System.out.println( TAG+" "+" MeanAndStdCalculator - trainDataDatasetHolder.shape().shape()[2] -  "+ trainDataDatasetHolder.shape()[2]);

//        INDArray testSlice = trainDataDatasetHolder.slice(0l, 0);
//        System.out.println( TAG+" "+" MeanAndStdCalculator - slice 0, 0 ---  ");
//        System.out.println( TAG+" "+" MeanAndStdCalculator - testSlice.shape().length -  "+ testSlice.shape().length);
//        System.out.println( TAG+" "+" MeanAndStdCalculator - testSlice.shape().shape()[0] -  "+ testSlice.shape()[0]);
//        System.out.println( TAG+" "+" MeanAndStdCalculator - testSlice.shape().shape()[1] -  "+ testSlice.shape()[1]);
////        System.out.println( TAG+" "+" MeanAndStdCalculator - testSlice.shape().shape()[2] -  "+ testSlice.shape()[2]);
//        INDArray testSlice2 = trainDataDatasetHolder.slice(0, 1);
//        System.out.println( TAG+" "+" MeanAndStdCalculator - slice 0, 1 ---  ");
//        System.out.println( TAG+" "+" MeanAndStdCalculator - testSlice2.shape().length -  "+ testSlice2.shape().length);
//        System.out.println( TAG+" "+" MeanAndStdCalculator - testSlice2.shape().shape()[0] -  "+ testSlice2.shape()[0]);
//        System.out.println( TAG+" "+" MeanAndStdCalculator - testSlice2.shape().shape()[1] -  "+ testSlice2.shape()[1]);
////        System.out.println( TAG+" "+" MeanAndStdCalculator - testSlice2.shape().shape()[2] -  "+ testSlice2.shape()[2]);
//        INDArray testSlice3 = trainDataDatasetHolder.slice(1, 1);
//        System.out.println( TAG+" "+" MeanAndStdCalculator - slice 1, 1 ---  ");
//        System.out.println( TAG+" "+" MeanAndStdCalculator - testSlice3.shape().length -  "+ testSlice3.shape().length);
//        System.out.println( TAG+" "+" MeanAndStdCalculator - testSlice3.shape().shape()[0] -  "+ testSlice3.shape()[0]);
//        System.out.println( TAG+" "+" MeanAndStdCalculator - testSlice3.shape().shape()[1] -  "+ testSlice3.shape()[1]);
////        System.out.println( TAG+" "+" MeanAndStdCalculator - testSlice3.shape().shape()[2] -  "+ testSlice3.shape()[2]);
//        INDArray testSlice4 = trainDataDatasetHolder.slice(1l, 2);
//        System.out.println( TAG+" "+" MeanAndStdCalculator - slice 1, 2 ---  ");
//        System.out.println( TAG+" "+" MeanAndStdCalculator - testSlice4.shape().length -  "+ testSlice4.shape().length);
//        System.out.println( TAG+" "+" MeanAndStdCalculator - testSlice4.shape().shape()[0] -  "+ testSlice4.shape()[0]);
//        System.out.println( TAG+" "+" MeanAndStdCalculator - testSlice4.shape().shape()[1] -  "+ testSlice4.shape()[1]);
////        System.out.println( TAG+" "+" MeanAndStdCalculator - testSlice4.shape().shape()[2] -  "+ testSlice4.shape()[2]);
//        INDArray testSlice5 = trainDataDatasetHolder.slice(2l, 2);
//        System.out.println( TAG+" "+" MeanAndStdCalculator - slice 2, 2 ---  ");
//        System.out.println( TAG+" "+" MeanAndStdCalculator - testSlice5.shape().length -  "+ testSlice5.shape().length);
//        System.out.println( TAG+" "+" MeanAndStdCalculator - testSlice5.shape().shape()[0] -  "+ testSlice5.shape()[0]);
//        System.out.println( TAG+" "+" MeanAndStdCalculator - testSlice5.shape().shape()[1] -  "+ testSlice5.shape()[1]);
////        System.out.println( TAG+" "+" MeanAndStdCalculator - testSlice5.shape().shape()[2] -  "+ testSlice5.shape()[2]);
//        INDArray testSlice6 = trainDataDatasetHolder.slice(0l, 2);
//        System.out.println( TAG+" "+" MeanAndStdCalculator - slice 0, 2 ---  ");
//        System.out.println( TAG+" "+" MeanAndStdCalculator - testSlice6.shape().length -  "+ testSlice6.shape().length);
//        System.out.println( TAG+" "+" MeanAndStdCalculator - testSlice6.shape().shape()[0] -  "+ testSlice6.shape()[0]);
//        System.out.println( TAG+" "+" MeanAndStdCalculator - testSlice6.shape().shape()[1] -  "+ testSlice6.shape()[1]);
////        System.out.println( TAG+" "+" MeanAndStdCalculator - testSlice6.shape().shape()[2] -  "+ testSlice6.shape()[2]);


//
//        // Slice the 'src' part of the train_dataset
//        INDArray srcSlice = trainDataDatasetHolder.slice(0, 0)
//                .slice(1, 1);
////                .slice(2, 2);
//
//        // Slice the 'trg' part of the trainDataDatasetHolder
//        INDArray trgSlice = trainDataDatasetHolder.slice(0L, 0);
////                .slice(new int[]{0, 1}, trainDataDatasetHolder.size(1))
////                .slice(2, 2);
//
//        System.out.println( TAG+" "+" MeanAndStdCalculator - srcSlice.shape().length -  "+ srcSlice.shape().length);
//        System.out.println( TAG+" "+" MeanAndStdCalculator - srcSlice.shape().shape()[0] -  "+ srcSlice.shape()[0]);
////        System.out.println( TAG+" "+" MeanAndStdCalculator - srcSlice.shape().shape()[1] -  "+ srcSlice.shape()[1]);
////        System.out.println( TAG+" "+" MeanAndStdCalculator - srcSlice.shape().shape()[2] -  "+ srcSlice.shape()[2]);
//
//        System.out.println( TAG+" "+" MeanAndStdCalculator - trgSlice.shape().length -  "+ trgSlice.shape().length);
//        System.out.println( TAG+" "+" MeanAndStdCalculator - trgSlice.shape().shape()[0] -  "+ trgSlice.shape()[0]);
//        System.out.println( TAG+" "+" MeanAndStdCalculator - trgSlice.shape().shape()[1] -  "+ trgSlice.shape()[1]);
//        System.out.println( TAG+" "+" MeanAndStdCalculator - trgSlice.shape().shape()[2] -  "+ trgSlice.shape()[2]);

        // Calculate the mean and standard deviation of concatenated tensors
//        INDArray datasetArray1 = trainData.next().getFeatures().get(all(), all(), interval(0, trainData.next().getFeatures().size(1)), interval(1, trainData.next().getFeatures().size(2)));
//        INDArray datasetArray2 = trainData.next().getFeatures().get(all(), all(), all(), interval(1, trainData.next().getFeatures().size(2)));
//        INDArray datasetArray1 = trainDataDatasetHolder.get(all(), all(), interval(0, trainDataDatasetHolder.size(1)), interval(1, trainDataDatasetHolder.size(2)));
//        INDArray datasetArray2 = trainDataDatasetHolder.get(all(), all(), all(), interval(1, trainDataDatasetHolder.size(2)));
//        INDArray datasetArray1 = trainDataDatasetHolder.get(all(), all(), interval(0, 1), interval(1, 2));
//        INDArray datasetArray2 = trainDataDatasetHolder.get(all(), all(), all(), interval(1, 2));
        INDArray datasetArray1 = trainDataDatasetHolder.get(all(), all(), interval(0, 1), interval(1, 2));
        INDArray datasetArray2 = trainDataDatasetHolder.get(all(), all(), interval(1, 2), interval(1, 2));
        System.out.println( TAG+" "+" MeanAndStdCalculator - datasetArray1.shape().length -  "+ datasetArray1.shape().length);
        System.out.println( TAG+" "+" MeanAndStdCalculator - datasetArray2.shape().length -  "+ datasetArray2.shape().length);
        System.out.println( TAG+" "+" MeanAndStdCalculator - datasetArray1.shape().shape()[0] -  "+ datasetArray1.shape()[0]);
        System.out.println( TAG+" "+" MeanAndStdCalculator - datasetArray1.shape().shape()[1] -  "+ datasetArray1.shape()[1]);
        System.out.println( TAG+" "+" MeanAndStdCalculator - datasetArray1.shape().shape()[2] -  "+ datasetArray1.shape()[2]);
        System.out.println( TAG+" "+" MeanAndStdCalculator - datasetArray2.shape().shape()[0] -  "+ datasetArray2.shape()[0]);
        System.out.println( TAG+" "+" MeanAndStdCalculator - datasetArray2.shape().shape()[1] -  "+ datasetArray2.shape()[1]);
        System.out.println( TAG+" "+" MeanAndStdCalculator - datasetArray2.shape().shape()[2] -  "+ datasetArray2.shape()[2]);

        INDArray concatenatedData = Nd4j.concat(1, datasetArray1, datasetArray2);
//        INDArray concatenatedData = Nd4j.concat(1,
//                trainData.next().getFeatures().get(all(), all(), interval(0, trainData.next().getFeatures().size(1)), interval(1, trainData.next().getFeatures().size(2))),
//                trainData.next().getFeatures().get(all(), all(), all(), interval(1, trainData.next().getFeatures().size(2)))
//        );
        INDArray meanArray = concatenatedData.mean(0, 1);
        INDArray stdArray = concatenatedData.std(0, 1);

        // Calculate the means and standard deviations for each dataset
        List<INDArray> means = new ArrayList<>();
        List<INDArray> stds = new ArrayList<>();
//        int[] uniqueDatasets = Nd4j.unique(trainData.next().getFeatures()).toIntVector();

        int index = 1;
//        trainData.reset();
        while (trainData.hasNext())
//            for (int i : uniqueDatasets)
            {
                trainDataDatasetHolder = trainData.next().getFeatures();
                
                INDArray datasetConcatenatedData = Nd4j.concat(1,
                        trainDataDatasetHolder.get( all(), interval(0, 1), interval(1, 2)),
                        trainDataDatasetHolder.get( all(), interval(0, 1), interval(1, 2))
//                        trainDataDatasetHolder.get( all(), interval(0, trainDataDatasetHolder.size(1)), interval(1, trainDataDatasetHolder.size(2))),
//                        trainDataDatasetHolder.get( all(), all(), interval(1, trainDataDatasetHolder.size(2)))
                );
                means.add(datasetConcatenatedData.mean(0, 1));
                stds.add(datasetConcatenatedData.std(0, 1));

                ++index;
            }

// Calculate the overall mean and standard deviation
        meanArray = Nd4j.vstack(means).mean(0);
        stdArray = Nd4j.vstack(stds).mean(0);

        mean = sd.var("mean"+" - "+mRandomNumericalId, meanArray);
        std = sd.var("std"+" - "+mRandomNumericalId, stdArray);
//        sd.associateArrayWithVariable(meanArray, mean);
//        sd.associateArrayWithVariable(stdArray, std);

        System.out.println( TAG+" "+" MeanAndStdCalculator - mean.eval() -  "+ mean.eval());
        System.out.println( TAG+" "+" MeanAndStdCalculator - std.eval() -  "+ std.eval());

    }


    private static void getEncoderInputDecoderInputAndDecoderMasks()
    {

        mRandom = new Random();
        mRandomNumericalId = mRandom.nextInt(10000000);

        INDArray trainDataDatasetHolder = trainData.next().getFeatures();
        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - trainDataDatasetHolder.shape().length -  "+ trainDataDatasetHolder.shape().length);
        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - trainDataDatasetHolder.shape()[0] -  "+ trainDataDatasetHolder.shape()[0]);
        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - trainDataDatasetHolder.shape()[1] -  "+ trainDataDatasetHolder.shape()[1]);
        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - trainDataDatasetHolder.shape()[2] -  "+ trainDataDatasetHolder.shape()[2]);
        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - trainDataDatasetHolder -  "+ trainDataDatasetHolder);


        HashMap<String, INDArray> placeholderData = new HashMap<>();
        trainData.reset();
        placeholderData.put("input",  trainData.next().getFeatures());
        placeholderData.put("label", trainData.next().getLabels());
        trainData.reset();

// Get the slice of 'src' array from dimensions [:, 1:, 2:4]  -- if using mean and std
        encInput = sd.var("encInput"+" - "+mRandomNumericalId, input.eval(placeholderData).get(all(), all(), all()));
//        encInput = sd.var("encInput"+" - "+mRandomNumericalId, trainDataDatasetHolder.get(all(), all(), all()));
//        encInput = sd.var("encInput", trainDataDatasetHolder.get(all(), interval(0, 1), interval(1, 2)));
//        encInput = sd.var("encInput", trainData.next().getFeatures().get(all(), interval(1L, trainData.next().getFeatures().size(1)), interval(2, 4)));
//        INDArray encInput = src.get(all(), interval(1, src.size(1)), interval(2, 4));

// Perform element-wise subtraction and division
//        encInput = encInput.sub(mean).div(std);

        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - encInput.getShape().toString() -  "+ Arrays.toString(encInput.getShape()));
        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - encInput.eval() -  "+ encInput.eval());


        //interval(0, -1) is the same as all()
        INDArray temp = trainDataDatasetHolder.get(all(), interval(0, -1), all());
//        INDArray temp = trainDataDatasetHolder.get(all(), interval(0, -1), interval(2, 3));
//        INDArray temp = trainDataDatasetHolder.get(all(), interval(0, -1), interval(1, 2));
        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - temp.shape().length -  "+ temp.shape().length);
        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - temp.shape()[0] -  "+ temp.shape()[0]);
        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - temp.shape()[1] -  "+ temp.shape()[1]);
        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - temp.shape()[2] -  "+ temp.shape()[2]);
        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - temp -  "+ temp);

// Get the 'trg' slice from the data array and remove the first two elements (lat, lon) in the second dimension
//        SDVariable target = sd.var("target"+" - "+mRandomNumericalId, input.eval(placeholderData).get(all(), interval(2, -1), all()));
        SDVariable target = sd.var("target"+" - "+mRandomNumericalId, input.eval(placeholderData).get(all(), all(), all()));
//        SDVariable target = sd.var("target"+" - "+mRandomNumericalId, trainDataDatasetHolder.get(all(), interval(0, 6), all()));
//        SDVariable target = sd.var("target", trainDataDatasetHolder.get(all(), interval(0, 5), all()));   //MAY NEED TO GO BACK TO THIS
//        SDVariable target = sd.var("target", trainDataDatasetHolder.get(all(), interval(0, -1), interval(1, 2)));
//        SDVariable target = sd.var("target", trainDataDatasetHolder.get(all(), interval(0, -1), interval(1, 2))).sub(mean).div(std);
//        SDVariable target = sd.var("target", trainData.next().getFeatures().get(all(), interval(0, -1), interval(2, 4))).sub(mean).div(std);

//        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - mean.shape().getShape().toString() -  "+ Arrays.toString(mean.getShape()));
//        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - std.shape().getShape().toString() -  "+ Arrays.toString(std.getShape()));
        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - target.getShape().toString() - 0 -  "+ Arrays.toString(target.getShape()));
        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - target.eval() - 0 -  "+ target.eval());

// Subtract mean from target and divide by std
//        target = target.sub(mean).div(std);


// Get the shape of the 'target' array
        long[] targetShape = target.getShape();

// Create a new array of zeros with an extra dimension of size 1
        long[] newShape = {targetShape[0], targetShape[1], 1};
        INDArray targetAppendArray = Nd4j.zeros(newShape);
        SDVariable targetAppend = sd.var("targetAppend"+" - "+mRandomNumericalId, targetAppendArray);
        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - targetAppend.getShape().toString() - 0 -  "+ Arrays.toString(targetAppend.getShape()));
        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - targetAppend.eval() - 0 -  "+ targetAppend.eval());

// Get the size of the dimension along which you want to concatenate
//        int axis = target.getShape().length - 1; // Assuming concatenating along the last dimension
//        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - axis -  "+ axis);

// Perform the concatenation

        target = sd.concat(2, target, targetAppend);
//    INDArray targetArray = Nd4j.concat(axis, target.getArr(), targetAppend);
        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - target.getShape().toString() - 1 -  "+ Arrays.toString(target.getShape()));
        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - target.eval() - 1 -  "+ target.eval());

//        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - target.getArr().shape().length 0-  "+ target.getArr().shape().length);
//        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - target.getArr().shape()[0] 0-  "+ target.getArr().shape()[0]);
//        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - target.getArr().shape()[1] 0-  "+ target.getArr().shape()[1]);
//        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - target.getArr().shape()[2] 0-  "+ target.getArr().shape()[2]);
//
//        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - targetAppend.getShape().length 0-  "+ targetAppend.getShape().length);
//        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - targetAppend.getShape()[0] 0-  "+ targetAppend.getShape()[0]);
//        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - targetAppend.getShape()[1] 0-  "+ targetAppend.getShape()[1]);
//        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - targetAppend.getShape()[2] 0-  "+ targetAppend.getShape()[2]);

//        target = sd.var(targetArray);
////    sd.associateArrayWithVariable(targetArray, target);
//        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - target.getShape().toString() - 1 -  "+ Arrays.toString(target.getShape()));

        ////////////////////////////////////////////////////////////
//        int[] startOfSeqData = {0, 0, 1};
//        targetShape = target.getShape();
//
//        INDArray startOfSeqArray = Nd4j.create(new double[]{0, 0, 1},new int[]{1, 3});
////        INDArray startOfSeqArray = Nd4j.create(startOfSeqData);
//        SDVariable startOfSeq = sd.var("startOfSeq", startOfSeqArray);
//        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - Arrays.toString(startOfSeq.getShape()) -0  "+ Arrays.toString(startOfSeq.getShape()));
//        SDVariable startOfSeqReshaped = sd.reshape(startOfSeq, startOfSeq.getShape()[0], startOfSeq.getShape()[1], 1);
////        startOfSeq = sd.expandDims(startOfSeq, 2);
////        startOfSeq = sd.expandDims(startOfSeq, 1);
////        startOfSeq = startOfSeq.reshape('c', 1, 1, startOfSeqData.length);
//        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - Arrays.toString(startOfSeqReshaped.getShape()) -1  "+ Arrays.toString(startOfSeqReshaped.getShape()));
//        SDVariable startOfSeqFinal = sd.tile("startOfSeqFinal",startOfSeqReshaped, 32, 1, 1);
////        SDVariable startOfSeqFinal = sd.repeat( startOfSeqReshaped, sd.var(startOfSeqArray), 0);
////        sd.repeat( startOfSeq, sd.var(startOfSeqArray), 0);
//        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - Arrays.toString(startOfSeqFinal.getShape()) -2  "+ Arrays.toString(startOfSeqFinal.getShape()));
//        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - startOfSeqArray.shape().length -  "+ startOfSeqArray.shape().length);
//        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - startOfSeqArray.shape()[0] -  "+ startOfSeqArray.shape()[0]);
//        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - startOfSeqArray.shape()[1] -  "+ startOfSeqArray.shape()[1]);
////        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - startOfSeq.shape()[2] -  "+ startOfSeqArray.shape()[2]);
//
//
////// Create a 1x3 INDArray representing the start_of_seq: [0, 0, 1]
////        SDVariable startOfSeq = sd.var("startOfSeq", Nd4j.create(new double[]{0, 0, 1}, new int[]{1, 3}));
////
////// Reshape start_of_seq to a 3D array of shape: [1, 1, 3]
////        startOfSeq = startOfSeq.reshape('c', 1, 1, 3);
////
////// Repeat start_of_seq along the first axis to match the number of sequences in the `target` array
////        int numSequences = (int) target.getShape()[0];
////        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - numSequences -  "+ numSequences);
////        INDArray startOfSeqArray = startOfSeq.getArr();
////        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - startOfSeqArray.shape().length 0-  "+ startOfSeqArray.shape().length);
////        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - startOfSeqArray.shape()[0] 0-  "+ startOfSeqArray.shape()[0]);
////        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - startOfSeqArray.shape()[1] 0-  "+ startOfSeqArray.shape()[1]);
////        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - startOfSeqArray.shape()[2] 0-  "+ startOfSeqArray.shape()[2]);
////
////        startOfSeqArray = startOfSeqArray.repeat(1, numSequences);
////        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - startOfSeqArray.shape().length 1-  "+ startOfSeqArray.shape().length);
////        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - startOfSeqArray.shape()[0] 1-  "+ startOfSeqArray.shape()[0]);
////        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - startOfSeqArray.shape()[1] 1-  "+ startOfSeqArray.shape()[1]);
////        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - startOfSeqArray.shape()[2] 1-  "+ startOfSeqArray.shape()[2]);
////
//////        startOfSeqArray = startOfSeqArray.repeat(numSequences, 1);
////        startOfSeq = sd.var(startOfSeqArray);
//////        sd.associateArrayWithVariable(startOfSeqArray, startOfSeq);
////
////// The shape of startOfSeq will be [target.shape[0], 1, 3]
////// You can access its values as follows:
//////double[][][] startOfSeqValues = startOfSeq.toDoubleMatrix();
////
////        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - Arrays.toString(startOfSeqArray.shape()) -  "+ Arrays.toString(startOfSeqArray.shape()));
////        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - Arrays.toString(targetArray.shape()) -  "+ Arrays.toString(targetArray.shape()));
////
////        startOfSeqArray = startOfSeqArray.reshape(32,3,1);
////        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - Arrays.toString(startOfSeqArray.shape()) -  "+ Arrays.toString(startOfSeqArray.shape()));
//
//        startOfSeqFinal.setDataType(DataType.FLOAT);
//        target.setDataType(DataType.FLOAT);
//        decInput.setDataType(DataType.FLOAT);
////        startOfSeqFinal.setDataType(DataType.DOUBLE);
////        target.setDataType(DataType.DOUBLE);
////        decInput.setDataType(DataType.DOUBLE);
//        decInput = sd.concat(0, startOfSeqFinal, target);
        ////////////////////////////////////////////////////////////


//            target = (data['trg'][:,:-1,2:4]
//        SDVariable target2 = sd.var("target2"+" - "+mRandomNumericalId, input.eval(placeholderData).get(all(), interval(2, -1), all()));
        SDVariable target2 = sd.var("target2"+" - "+mRandomNumericalId, trainDataDatasetHolder.get(all(), interval(0, 6), all()));
//        SDVariable target2 = sd.var("target2", trainDataDatasetHolder.get(all(), interval(0, 5), all()));   //MAY NEED TO GO BACK TO THIS
        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - target2.eval() -0  "+ target2.eval());

//            target_append = torch.zeros((target.shape[0],target.shape[1],1))
        SDVariable targetAppend2 = sd.var("targetAppend2"+" - "+mRandomNumericalId, Nd4j.zeros(target2.getShape()[0],target2.getShape()[1],1));
        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - targetAppend2.eval() -0  "+ targetAppend2.eval());

//            target = torch.cat((target,target_append),-1)
        target2 = sd.concat(-1, target2, targetAppend2);
        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - target2.eval().shapeInfoToString -0  "+ target2.eval().shapeInfoToString());
        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - target2.eval() -1  "+ target2.eval());

//            start_of_seq = torch.Tensor([0, 0, 1]).unsqueeze(0).unsqueeze(1).repeat(target.shape[0],1,1)
        INDArray startOfSeqArray = Nd4j.create(new double[]{0, 0, 0, 0, 0, 1},new int[]{1, 6});
//        INDArray startOfSeqArray = Nd4j.create(new double[]{0, 0, 0, 0, 1},new int[]{1, 5});   //MAY NEED TO GO BACK TO THIS
        SDVariable startSeqStart = sd.var("startSeqStart"+" - "+mRandomNumericalId, startOfSeqArray);
        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - startSeqStart.getShape()[0]  "+ startSeqStart.getShape()[0]);
        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - startSeqStart.getShape()[1]  "+ startSeqStart.getShape()[1]);
        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - startSeqStart.eval() -0  "+ startSeqStart.eval());

        SDVariable startOfSeqStartReshaped = sd.reshape(startSeqStart, startSeqStart.getShape()[0], startSeqStart.getShape()[1], 1);
        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - startOfSeqStartReshaped.eval().shapeInfoToString -0  "+ startOfSeqStartReshaped.eval().shapeInfoToString());
        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - startOfSeqStartReshaped.eval() -0  "+ startOfSeqStartReshaped.eval());

        SDVariable startOfSeq = sd.tile("startOfSeq"+" - "+mRandomNumericalId, startOfSeqStartReshaped, batch_size, 1, 1);
//        SDVariable startOfSeq = sd.tile("startOfSeq"+" - "+mRandomNumericalId, startOfSeqStartReshaped, 32, 1, 1);
        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - startOfSeq.eval().shapeInfoToString -0  "+ startOfSeq.eval().shapeInfoToString());
        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - startOfSeq.eval() -0  "+ startOfSeq.eval());

        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - Arrays.toString(startOfSeq.getShape()) -2  "+ Arrays.toString(startOfSeq.getShape()));
        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - Arrays.toString(target2.getShape()) -2  "+ Arrays.toString(target2.getShape()));

        startOfSeq.setDataType(DataType.FLOAT);
        target2.setDataType(DataType.FLOAT);
        decInput.setDataType(DataType.FLOAT);
//            dec_input = torch.cat((start_of_seq, target), 1)
        decInput = sd.concat(-1, startOfSeq, target2);

//        decInput = sd.concat(1, startOfSeqFinal, target);
        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - decInput.eval().shapeInfoToString -0  "+ decInput.eval().shapeInfoToString());
        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - decInput.eval() 0-  "+ decInput.eval());

//        INDArray decInputArray = Nd4j.concat(1, startOfSeqArray, targetArray);
//        decInput = sd.var("decInput", decInputArray);
        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - Arrays.toString(decInput.getShape()) -1  "+ Arrays.toString(decInput.getShape()));


        INDArray decSourceMaskArray = Nd4j.ones(encInput.getShape()[0], 1, encInput.getShape()[1]);
        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - decSourceMaskArray.shape().length 1-  "+ decSourceMaskArray.shape().length);
        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - decSourceMaskArray.shape()[0] 1-  "+ decSourceMaskArray.shape()[0]);
        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - decSourceMaskArray.shape()[1] 1-  "+ decSourceMaskArray.shape()[1]);
        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - decSourceMaskArray.shape()[2] 1-  "+ decSourceMaskArray.shape()[2]);
        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - decSourceMaskArray 1-  "+ decSourceMaskArray);


        int decInputSizeAlongDimension1 = (int) decInput.eval().shape()[1];
//        int decInputSizeAlongDimension1 = (int) decInput.getShape()[1];
        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - decInputSizeAlongDimension1 -  "+ decInputSizeAlongDimension1);
        SDVariable decTargetMaskStarter = TransformerArchitectureModel.subsequent_mask(decInputSizeAlongDimension1);
        decInputMask = sd.tile("decInputMask"+" - "+mRandomNumericalId, decTargetMaskStarter, (int) decInput.eval().shape()[0], 1, 1);
        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - decInputMask.eval() -  "+ decInputMask.eval());

//        INDArray decTargetMaskArray = TransformerArchitectureModel.subsequent_mask(decInputSizeAlongDimension1).getArr();
//        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - decTargetMaskArray.shape().length 1-  "+ decTargetMaskArray.shape().length);
//        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - decTargetMaskArray.shape()[0] 1-  "+ decTargetMaskArray.shape()[0]);
//        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - decTargetMaskArray.shape()[1] 1-  "+ decTargetMaskArray.shape()[1]);
//        System.out.println( TAG+" "+" getEncoderInputDecoderInputAndDecoderMasks - decTargetMaskArray.shape()[2] 1-  "+ decTargetMaskArray.shape()[2]);
////        INDArray decTargetMaskArray = TransformerArchitectureModel.subsequent_mask(decInputSizeAlongDimension1).getArr().repeat((int) decInput.getShape()[0], 1, 1);
//        //.repeat((int) decInput.getShape()[0], 1, 1)

        decSourceMask = sd.var("decSourceMask"+" - "+mRandomNumericalId, decSourceMaskArray);
//        decInputMask = sd.var("decInputMask", decTargetMaskArray);
//        sd.associateArrayWithVariable(decSourceMaskArray, decSourceMask);
//        sd.associateArrayWithVariable(decTargetMaskArray, decInputMask);
    }

    //This method downloads the data, and converts the "one time series per line" format into a suitable
    //CSV sequence format that DataVec (CsvSequenceRecordReader) and DL4J can read.
    private static void downloadUCIData() throws Exception {
        if (baseDir.exists()) return;    //Data already exists, don't download it again

//      String gpsTrackPointsFilePath = "src/main/assets/latLonTimeDataset.csv";  //DO NOT USE DUPLICATE OF neuralNetworkDataSet.csv WITHOUT ADDITIONAL COLUMNS
//        String gpsTrackPointsFilePath = "src/main/assets/paths.csv";
        String gpsTrackPointsFilePath = "src/main/assets/neuralNetworkDataSet.csv";
//        String gpsTrackPointsFilePath = "src/main/assets/go_track_trackspoints_modified-2.csv";  //DO NOT USE. CONTAINS TEST DATA NOT GENERATED BY ANDROID APP
        InputStream myInput = null;
//        myInput = new FileInputStream(gpsTrackPointsFilePath);
        InputStream myInput2 = null;
        myInput2 = new FileInputStream(gpsTrackPointsFilePath);

        File myFile = new File(gpsTrackPointsFilePath);

        Path myPath = myFile.toPath();

//        List<String[]> recordsDateTimeChangedToDouble = new ArrayList<String[]>();
        List<String[]> recordsDateTimeChangedToDouble2 = new ArrayList<String[]>();

//        recordsDateTimeChangedToDouble = readLineByLine(myPath);
        recordsDateTimeChangedToDouble2 = readLineByLine2(myPath);

        String pathToFileWithDateTimeChangedToDouble = "src/main/assets/go_track_trackspoints_modified-3.csv";
        String pathToFileWithDateTimeChangedToDouble2 = "src/main/assets/go_track_trackspoints_modified-4.csv";

//        writeAllLines(recordsDateTimeChangedToDouble, pathToFileWithDateTimeChangedToDouble);
        writeAllLines(recordsDateTimeChangedToDouble2, pathToFileWithDateTimeChangedToDouble2);

//        myInput = new FileInputStream(pathToFileWithDateTimeChangedToDouble);
        myInput2 = new FileInputStream(pathToFileWithDateTimeChangedToDouble2);

//        String data = IOUtils.toString(myInput, (Charset) null);
        String data2 = IOUtils.toString(myInput2, (Charset) null);

//        String[] lines = data.split("\n");
        String[] lines2 = data2.split("\n");

//        int lines2Length = lines2.length;
//        for(int i = 0; i < lines2Length; ++i)
//        {
//            Log.info(" - lines2[i] - "+lines2[i]);
//        }

        //Create directories
        baseDir.mkdir();
        baseTrainDir.mkdir();
        featuresDirTrain.mkdir();
        labelsDirTrain.mkdir();
        baseTestDir.mkdir();
        featuresDirTest.mkdir();
        labelsDirTest.mkdir();

        int lineCount = 0;
//        List<Pair<String, Integer>> contentAndLabels = new ArrayList<>();
        List<Pair<String, double[]>> contentAndLabels2 = new ArrayList<Pair<String, double[]>>();

//        List<Pair<String, Long>> contentAndLabels2 = new ArrayList<>();

//        for (String line : lines) {
//
//            if(lineCount == 0)
//            {
//                lineCount++;
//                continue;
//            }
//
//            String transposed = line.replaceAll(" +", "\n");
//
//            String comma = ",";
//            int commaLastIndex = line.lastIndexOf(comma);
//            String contentAndLabelsTrackIdKeyString = "";
//            for(int k = commaLastIndex - 1; k > 0; --k)
//            {
//                if(line.charAt(k) != ',') {
//                    contentAndLabelsTrackIdKeyString = line.charAt(k) + contentAndLabelsTrackIdKeyString;
//                }
//                else {
//                    break;
//                }
//            }
//
//            int contentAndLabelsKey = (int)Double.parseDouble(contentAndLabelsTrackIdKeyString);
//            contentAndLabels.add(new Pair<>(transposed, contentAndLabelsKey));
//         }

        //USING GEOHASH - START - %~%%~%%~%%~%%~%%~%%~%%~%%~%%~%%~%%~%%~%%~%%~%%~%%~%%~%%~%%~%%~%%~%%~%%~%%~%%~%%~%%~%%~%%~%

        for (String line : lines2) {

//            System.out.println( TAG+" "+" - line - 0 - "+line);

            if(lineCount == 0)
            {
                lineCount++;
                continue;
            }

            String transposed = line.replaceAll(" +", "\n");

            String contentAndLabelsGeohashKeyString = "";
            for(int k = line.length() - 1; k > 0; --k)
            {
                if(line.charAt(k) != ',') {
                    contentAndLabelsGeohashKeyString = line.charAt(k) + contentAndLabelsGeohashKeyString;
                }
                else {
                    break;
                }
            }
//            System.out.println( TAG+" "+" - contentAndLabelsGeohashKeyString - 0 - "+contentAndLabelsGeohashKeyString);

            String comma = ",";
            int commaLastIndex = transposed.lastIndexOf(comma);
//            System.out.println( TAG+" "+"transposed - 0 - "+transposed);
            String temp = transposed.substring(0, commaLastIndex);
            commaLastIndex = temp.lastIndexOf(comma);
            String temp2 = transposed.substring(0, commaLastIndex);
            String temp3 = temp2.substring(temp2.indexOf(comma) +1);
//            System.out.println( TAG+" "+"temp3 -  "+temp3);
            transposed = "";
            transposed = temp;
//            transposed = temp2;
//            transposed = temp3;
//            System.out.println( TAG+" "+"transposed - 1 - "+transposed);
            temp = "";

//            GeoPoint predictedOutputHashString = GeoHashUtils.decode(Long.parseLong(contentAndLabelsGeohashKeyString));
////          GeoPoint predictedOutputHashString = GeoHashUtils.decode(GeoHashUtils.encodeAsLong(Double.parseDouble(lineIn[1]),Double.parseDouble(lineIn[2]), 8));
//            double latitude = predictedOutputHashString.getLat();
//            double longitude = predictedOutputHashString.getLon();
//
//            Log.info(" - contentAndLabelsGeohashKeyString - " + contentAndLabelsGeohashKeyString);
//            Log.info(" - latitude - " + latitude);
//            Log.info(" - longitude - " + longitude);

//            System.out.println( TAG+" "+"contentAndLabelsGeohashKeyString - "+contentAndLabelsGeohashKeyString);

//            Log.info(" - line - 0 - " + line);

            String latitudeString = "";
            String longitudeString = "";
//            System.out.println( TAG+" "+" - line.indexOf(\",\") - 0 - "+line.indexOf(","));
            for(int v = 0 ; v < line.length(); ++v)
            {
//                System.out.println( TAG+" "+" - line.charAt(v) - "+line.charAt(v));
                if(line.charAt(v) != ',')
                {
                    latitudeString = latitudeString + line.charAt(v);
                }
                else {
                    line = line.substring(line.indexOf(",") + 1);
//                    Log.info(" - line - 1 - " + line);
                    for (int w = 0; w < line.length(); ++w)
                    {
                        if (line.charAt(w) != ',') {
                            longitudeString = longitudeString + line.charAt(w);
                        } else {
                            break;
                        }
                    }
                    break;
                }

            }
//            System.out.println( TAG+" "+" - latitudeString - 0 - "+latitudeString);
//            System.out.println( TAG+" "+" - longitudeString - 0 - "+longitudeString);

            double LatitudeStringToDouble = Double.parseDouble(latitudeString);
            double LongitudeStringToDouble = Double.parseDouble(longitudeString);


            double lines2LatitudeLongitudeHolder[] = {LatitudeStringToDouble, LongitudeStringToDouble};
            contentAndLabels2.add(new Pair<>(transposed, lines2LatitudeLongitudeHolder));
        }

//        for(Pair contentAndLabels2Record: contentAndLabels2)
//        {
//            Log.info(" - contentAndLabels2Record.getFirst() - "+contentAndLabels2Record.getFirst());
//            Log.info(" - contentAndLabels2Record.getSecond() - "+contentAndLabels2Record.getSecond());
//        }

        //USING GEOHASH - END - %~%%~%%~%%~%%~%%~%%~%%~%%~%%~%%~%%~%%~%%~%%~%%~%%~%%~%%~%%~%%~%%~%%~%%~%%~%%~%%~%%~%%~%%~%


        contentAndLabels2Size = contentAndLabels2.size();
        System.out.println( TAG+" "+"contentAndLabels2Size - "+contentAndLabels2Size);

//        contentAndLabelsSize = contentAndLabels.size();
//        System.out.println( TAG+" "+"contentAndLabelsSize - "+contentAndLabelsSize);

//        //Randomize and do a train/test split:
//        Collections.shuffle(contentAndLabels2, new Random(12345));


//        Collections.sort(contentAndLabels, (one, another) -> {
//
//            int trackIdOne = one.getSecond();
//            int trackIdAnother = another.getSecond();
//
//            return (one.getSecond() < another.getSecond() ? -1 :
//                    (trackIdOne == trackIdAnother ? 0 : 1));
//
//        });

//            Collections.sort(contentAndLabels2, (one, another) -> {
//
//            Long trackIdOne = GeoHashUtils.encodeAsLong(one.getSecond()[0], one.getSecond()[1], 8);
//            Long trackIdAnother = GeoHashUtils.encodeAsLong(another.getSecond()[0], one.getSecond()[1], 8);
//
//                return (String.valueOf(GeoHashUtils.encodeAsLong(one.getSecond()[0], one.getSecond()[1], 8)).compareTo(String.valueOf(GeoHashUtils.encodeAsLong(another.getSecond()[0], one.getSecond()[1], 8))) < 0 ? -1 :
//                    (trackIdOne.equals(trackIdAnother) ? 0 : 1));
//
//        });

        int contentAndLabels2RecordCount = 0;
        for(Pair contentAndLabels2Record : contentAndLabels2)
        {
//                System.out.println( TAG+" "+" - contentAndLabels2Record.getFirst().toString() - "+contentAndLabels2Record.getFirst().toString());
//                System.out.println( TAG+" "+" - contentAndLabels2Record.getSecond().toString() - "+contentAndLabels2Record.getSecond().toString());
            contentAndLabels2RecordCount++;
            if(contentAndLabels2RecordCount > 1000)
            {
                break;
            }
        }

        nItemsInDataSet = contentAndLabels2Size;   //75% train, 25% test
//        nItemsInDataSet = contentAndLabelsSize;   //75% train, 25% test
        nTrain = (int)Math.round(nItemsInDataSet * .75);
        nTest = nItemsInDataSet - nTrain;
        int trainCount = 0;
        int testCount = 0;

        System.out.println( TAG+" "+"nTrain - "+nTrain);

        //USING GEOHASH - START - @~@@~@@~@@~@@~@@~@@~@@~@@~@@~@@~@@~@@~@@~@@~@@~@@~@@~@@~@@~@@~@@~@@~@@~@@~@@~@@~@@~@@~@@~@

        double[] labelHolder2 = contentAndLabels2.get(0).getSecond();

        String featureAllBelongingToOneLabelHolder2 = "";
        String featureHolderSingle2 = contentAndLabels2.get(0).getFirst();
        int featureHolderIndex2 = 0;
        int featureRecordIndexLocal = 0;
        ArrayList<double[]>  latitudeLongitudeOfRecordsWithTheSameGeohash = new ArrayList<double[]>();

        System.out.println( TAG+" "+"labelHolder2 - "+labelHolder2);
        System.out.println( TAG+" "+"featureHolderSingle2 - "+featureHolderSingle2);

        for (Pair<String, double[]> p : contentAndLabels2) {
            //Write output in a format we can read, in the appropriate locations
            File outPathFeatures;
            File outPathLabels;

//                    System.out.println( TAG+" "+"featureHolderIndex2 - "+featureHolderIndex2);

//TRAIN .CSV FILE CREATION - START - #**##**##**##**##**##**##**##**##**##**##**##**##**##**##**##**##**##**##**##**##**#
            if (featureHolderIndex2 < nTrain) {

//                System.out.println( TAG+" "+"labelHolder2 - "+labelHolder2);
//                System.out.println( TAG+" "+" - labelHolder2[0] - "+labelHolder2[0]);
//                System.out.println( TAG+" "+" - labelHolder2[1] - "+labelHolder2[1]);
                Long locationBeingCompared = GeoHashUtils.encodeAsLong(labelHolder2[0], labelHolder2[1], 8);
                Long locationToCompareToGeohash = GeoHashUtils.encodeAsLong(p.getSecond()[0], p.getSecond()[1], 8);
                if(String.valueOf(locationBeingCompared).equals(String.valueOf(locationToCompareToGeohash)))
                {
                    if(featureAllBelongingToOneLabelHolder2.equalsIgnoreCase(""))
                    {
                        featureAllBelongingToOneLabelHolder2 = p.getFirst();
                        ++featureRecordIndexLocal;
                        latitudeLongitudeOfRecordsWithTheSameGeohash.add(new double[]{p.getSecond()[0], p.getSecond()[1]});
                        String firstFeatureFileRecord = p.getFirst();
                        Log.info(" - firstFeatureFileRecord - 0 - "+firstFeatureFileRecord);
                    }
                    else
                    {
                        Log.info(" - featureRecordIndexLocal - "+featureRecordIndexLocal);
                        featureAllBelongingToOneLabelHolder2 = featureAllBelongingToOneLabelHolder2 + "\n" + p.getFirst();
                        ++featureRecordIndexLocal;
                        latitudeLongitudeOfRecordsWithTheSameGeohash.add(new double[]{p.getSecond()[0], p.getSecond()[1]});
                    }
                    featureHolderIndex2++;
                    continue;
                }
                else
                {
                    lastTrainCount = trainCount;
                    if(featureAllBelongingToOneLabelHolder2.equalsIgnoreCase(""))
                    {
                        featureAllBelongingToOneLabelHolder2 = contentAndLabels2.get(featureHolderIndex2 - 1).getFirst();

                        String firstFeatureFileRecord = contentAndLabels2.get(featureHolderIndex2 - 1).getFirst();
                        Log.info(" - firstFeatureFileRecord - 1 - "+firstFeatureFileRecord);
                    }
                    outPathFeatures = new File(featuresDirTrain, trainCount + ".csv");
                    outPathLabels = new File(labelsDirTrain, trainCount + ".csv");
                    trainCount++;
//                    System.out.println( TAG+" "+"featureAllBelongingToOneLabelHolder2 - "+featureAllBelongingToOneLabelHolder2);
                    FileUtils.writeStringToFile(outPathFeatures, featureAllBelongingToOneLabelHolder2, (Charset) null);

                    Log.info(" - labelHolder2[0] - 0 - "+labelHolder2[0]);
                    Log.info(" - labelHolder2[1] - 0 - "+labelHolder2[1]);
                    Log.info(" - contentAndLabels2.get(featureHolderIndex2 - 1).getSecond() - 1 - "+contentAndLabels2.get(featureHolderIndex2 - 1).getSecond()[0]);
                    Log.info(" - contentAndLabels2.get(featureHolderIndex2 - 1).getSecond() - 1 - "+contentAndLabels2.get(featureHolderIndex2 - 1).getSecond()[1]);
                    Log.info(" - featureAllBelongingToOneLabelHolder2 - "+featureAllBelongingToOneLabelHolder2);
//                    System.out.println( TAG+" "+"labelHolder2 - "+labelHolder2);
//                    System.out.println( TAG+" "+"String.valueOf(labelHolder2) - "+String.valueOf(labelHolder2));
//                    System.out.println( TAG+" "+" - GeoHashUtils.decode(labelHolder2).getLat() - "+GeoHashUtils.decode(labelHolder2).getLat());
//                    System.out.println( TAG+" "+" - GeoHashUtils.decode(labelHolder2).getLon() - "+GeoHashUtils.decode(labelHolder2).getLon());
                    Log.info(" - featureRecordIndexLocal - 0 - "+featureRecordIndexLocal);
                    if(featureRecordIndexLocal > 1)
                    {
                        double labelLat = GeoHashUtils.decode(GeoHashUtils.encodeAsLong(labelHolder2[0], labelHolder2[1], 8)).getLat();
                        double labelLon = GeoHashUtils.decode(GeoHashUtils.encodeAsLong(labelHolder2[0], labelHolder2[1], 8)).getLon();
                        Log.info(" - labelLat - 1 - "+labelLat);
                        Log.info(" - labelLon - 1 - "+labelLon);

                        double[] latitudeOfRecordsWithTheSameGeohashArray = new double[latitudeLongitudeOfRecordsWithTheSameGeohash.size()];
                        double[] longitudeOfRecordsWithTheSameGeohashArray = new double[latitudeLongitudeOfRecordsWithTheSameGeohash.size()];
                        for(int t = 0; t < latitudeLongitudeOfRecordsWithTheSameGeohash.size(); ++t)
                        {
                            latitudeOfRecordsWithTheSameGeohashArray[t] = latitudeLongitudeOfRecordsWithTheSameGeohash.get(t)[0];
                            longitudeOfRecordsWithTheSameGeohashArray[t] = latitudeLongitudeOfRecordsWithTheSameGeohash.get(t)[1];
                        }
                        double meanLatitudeOfRecordsWithTheSameGeohash = StatUtils.mean(latitudeOfRecordsWithTheSameGeohashArray);
                        double meanLongitudeOfRecordsWithTheSameGeohash = StatUtils.mean(longitudeOfRecordsWithTheSameGeohashArray);
                        double medianLatitudeOfRecordsWithTheSameGeohash = StatUtils.percentile(latitudeOfRecordsWithTheSameGeohashArray, 0.50);
                        double medianLongitudeOfRecordsWithTheSameGeohash = StatUtils.percentile(longitudeOfRecordsWithTheSameGeohashArray, 0.50);
                        Log.info(" - meanLatitudeOfRecordsWithTheSameGeohash - 1 - "+meanLatitudeOfRecordsWithTheSameGeohash);
                        Log.info(" - meanLongitudeOfRecordsWithTheSameGeohash - 1 - "+meanLongitudeOfRecordsWithTheSameGeohash);
                        Log.info(" - medianLatitudeOfRecordsWithTheSameGeohash - 1 - "+medianLatitudeOfRecordsWithTheSameGeohash);
                        Log.info(" - medianLongitudeOfRecordsWithTheSameGeohash - 1 - "+medianLongitudeOfRecordsWithTheSameGeohash);

//                        labelLat = meanLatitudeOfRecordsWithTheSameGeohash;
//                        labelLon = meanLongitudeOfRecordsWithTheSameGeohash;

                        latitudeLongitudeOfRecordsWithTheSameGeohash = new ArrayList<double[]>();

                        FileUtils.writeStringToFile(outPathLabels, labelLat+","+labelLon, (Charset) null);
                        featureRecordIndexLocal = 0;
                    }
                    else
                    {
                        featureRecordIndexLocal = 0;
                        FileUtils.writeStringToFile(outPathLabels, contentAndLabels2.get(featureHolderIndex2 - 1).getSecond()[0]+","+contentAndLabels2.get(featureHolderIndex2 - 1).getSecond()[1], (Charset) null);
//                    FileUtils.writeStringToFile(outPathLabels, labelHolder2[0]+","+labelHolder2[1], (Charset) null);
                    }

                    if(featureHolderIndex2 == nTrain - 1)
                    {

                        Log.info(" - Writing to file last record - 0 --- ");
                        FileUtils.writeStringToFile(outPathFeatures, contentAndLabels2.get(featureHolderIndex2).getFirst(), (Charset) null);
                        FileUtils.writeStringToFile(outPathLabels, contentAndLabels2.get(featureHolderIndex2).getSecond()[0]+","+contentAndLabels2.get(featureHolderIndex2).getSecond()[1], (Charset) null);

                    }
                    labelHolder2 = p.getSecond();
//                    featureAllBelongingToOneLabelHolder2 = "id,latitude,longitude,time,track_id,geohash";
                    featureAllBelongingToOneLabelHolder2 = "";
                    featureHolderIndex2++;
                    featureRecordIndexLocal = 0;

                }

            }
//TRAIN .CSV FILE CREATION - END - #**##**##**##**##**##**##**##**##**##**##**##**##**##**##**##**##**##**##**##**##**#

//TEST .CSV FILE CREATION - START - *_**_**_**_**_**_**_**_**_**_**_**_**_**_**_**_**_**_**_**_**_**_**_**_**_**_*
            else
            {
                Long locationBeingCompared = GeoHashUtils.encodeAsLong(labelHolder2[0], labelHolder2[1], 8);
                Long locationToCompareToGeohash = GeoHashUtils.encodeAsLong(p.getSecond()[0], p.getSecond()[1], 8);
                if(String.valueOf(locationBeingCompared).equals(String.valueOf(locationToCompareToGeohash)))
                {
                    if(featureAllBelongingToOneLabelHolder2.equalsIgnoreCase(""))
                    {
                        featureAllBelongingToOneLabelHolder2 = p.getFirst();
                        ++featureRecordIndexLocal;
                        latitudeLongitudeOfRecordsWithTheSameGeohash.add(new double[]{p.getSecond()[0], p.getSecond()[1]});
                        String firstFeatureFileRecord = p.getFirst();
                        Log.info(" - firstFeatureFileRecord - 2 - "+firstFeatureFileRecord);

                        String firstFeatureFileRecord2 = contentAndLabels2.get(featureHolderIndex2 - 1).getFirst();
                        Log.info(" - firstFeatureFileRecord2 - 2a - "+firstFeatureFileRecord2);

                    }
                    else
                    {
                        Log.info(" - featureRecordIndexLocal - "+featureRecordIndexLocal);
                        featureAllBelongingToOneLabelHolder2 = featureAllBelongingToOneLabelHolder2 + "\n" + p.getFirst();
                        ++featureRecordIndexLocal;
                        latitudeLongitudeOfRecordsWithTheSameGeohash.add(new double[]{p.getSecond()[0], p.getSecond()[1]});
                    }

                    featureHolderIndex2++;
                    continue;
                }
                else
                {
                    lastTestCount = testCount;
                    if(featureAllBelongingToOneLabelHolder2.equalsIgnoreCase(""))
                    {
                        featureAllBelongingToOneLabelHolder2 = contentAndLabels2.get(featureHolderIndex2 - 1).getFirst();

                        String firstFeatureFileRecord = contentAndLabels2.get(featureHolderIndex2 - 1).getFirst();
                        Log.info(" - firstFeatureFileRecord - 3 - "+firstFeatureFileRecord);
                    }
                    outPathFeatures = new File(featuresDirTest, testCount + ".csv");
                    outPathLabels = new File(labelsDirTest, testCount + ".csv");
                    testCount++;
                    Log.info(" - labelHolder2[0] - 1 - "+labelHolder2[0]);
                    Log.info(" - labelHolder2[1] - 1 - "+labelHolder2[1]);
                    Log.info(" - contentAndLabels2.get(featureHolderIndex2 - 1).getSecond() - 1 - "+contentAndLabels2.get(featureHolderIndex2 - 1).getSecond()[0]);
                    Log.info(" - contentAndLabels2.get(featureHolderIndex2 - 1).getSecond() - 1 - "+contentAndLabels2.get(featureHolderIndex2 - 1).getSecond()[1]);
                    Log.info(" - featureAllBelongingToOneLabelHolder2 - "+featureAllBelongingToOneLabelHolder2);

                    Log.info(" - featureRecordIndexLocal - 1 - "+featureRecordIndexLocal);
                    if(featureRecordIndexLocal > 1)
                    {
                        double labelLat = GeoHashUtils.decode(GeoHashUtils.encodeAsLong(labelHolder2[0], labelHolder2[1], 8)).getLat();
                        double labelLon = GeoHashUtils.decode(GeoHashUtils.encodeAsLong(labelHolder2[0], labelHolder2[1], 8)).getLon();
                        Log.info(" - labelLat - 1 - "+labelLat);
                        Log.info(" - labelLon - 1 - "+labelLon);

                        double[] latitudeOfRecordsWithTheSameGeohashArray = new double[latitudeLongitudeOfRecordsWithTheSameGeohash.size()];
                        double[] longitudeOfRecordsWithTheSameGeohashArray = new double[latitudeLongitudeOfRecordsWithTheSameGeohash.size()];
                        for(int t = 0; t < latitudeLongitudeOfRecordsWithTheSameGeohash.size(); ++t)
                        {
                            latitudeOfRecordsWithTheSameGeohashArray[t] = latitudeLongitudeOfRecordsWithTheSameGeohash.get(t)[0];
                            longitudeOfRecordsWithTheSameGeohashArray[t] = latitudeLongitudeOfRecordsWithTheSameGeohash.get(t)[1];
                        }
                        double meanLatitudeOfRecordsWithTheSameGeohash = StatUtils.mean(latitudeOfRecordsWithTheSameGeohashArray);
                        double meanLongitudeOfRecordsWithTheSameGeohash = StatUtils.mean(longitudeOfRecordsWithTheSameGeohashArray);
                        double medianLatitudeOfRecordsWithTheSameGeohash = StatUtils.percentile(latitudeOfRecordsWithTheSameGeohashArray, 0.50);
                        double medianLongitudeOfRecordsWithTheSameGeohash = StatUtils.percentile(longitudeOfRecordsWithTheSameGeohashArray, 0.50);
                        Log.info(" - meanLatitudeOfRecordsWithTheSameGeohash - 1 - "+meanLatitudeOfRecordsWithTheSameGeohash);
                        Log.info(" - meanLongitudeOfRecordsWithTheSameGeohash - 1 - "+meanLongitudeOfRecordsWithTheSameGeohash);
                        Log.info(" - medianLatitudeOfRecordsWithTheSameGeohash - 1 - "+medianLatitudeOfRecordsWithTheSameGeohash);
                        Log.info(" - medianLongitudeOfRecordsWithTheSameGeohash - 1 - "+medianLongitudeOfRecordsWithTheSameGeohash);

//                        labelLat = meanLatitudeOfRecordsWithTheSameGeohash;
//                        labelLon = meanLongitudeOfRecordsWithTheSameGeohash;

                        latitudeLongitudeOfRecordsWithTheSameGeohash = new ArrayList<double[]>();

                        FileUtils.writeStringToFile(outPathLabels, labelLat+","+labelLon, (Charset) null);
                        featureRecordIndexLocal = 0;
                    }
                    else
                    {
                        featureRecordIndexLocal = 0;
                        FileUtils.writeStringToFile(outPathLabels, contentAndLabels2.get(featureHolderIndex2 - 1).getSecond()[0]+","+contentAndLabels2.get(featureHolderIndex2 - 1).getSecond()[1], (Charset) null);
//                    FileUtils.writeStringToFile(outPathLabels, labelHolder2[0]+","+labelHolder2[1], (Charset) null);
                    }

                    FileUtils.writeStringToFile(outPathFeatures, featureAllBelongingToOneLabelHolder2, (Charset) null);

                    if(featureHolderIndex2 == contentAndLabels2.size() - 1)
                    {
                        Log.info(" - Writing to file last record - 1 --- ");
                        FileUtils.writeStringToFile(outPathFeatures, contentAndLabels2.get(featureHolderIndex2).getFirst(), (Charset) null);
                        FileUtils.writeStringToFile(outPathLabels, contentAndLabels2.get(featureHolderIndex2).getSecond()[0]+","+contentAndLabels2.get(featureHolderIndex2).getSecond()[1], (Charset) null);
                    }
                    labelHolder2 = p.getSecond();
                    featureAllBelongingToOneLabelHolder2 = "";
                    featureHolderIndex2++;
                    featureRecordIndexLocal = 0;
                }

            }
//TEST .CSV FILE CREATION - END - *_**_**_**_**_**_**_**_**_**_**_**_**_**_**_**_**_**_**_**_**_**_**_**_**_**_*

        }

//        USING GEOHASH - END - @~@@~@@~@@~@@~@@~@@~@@~@@~@@~@@~@@~@@~@@~@@~@@~@@~@@~@@~@@~@@~@@~@@~@@~@@~@@~@@~@@~@@~@@~@
    }

    public static List<String[]> readLineByLine(Path filePath) throws Exception {
        List<String[]> list = new ArrayList<>();
        try (Reader reader = Files.newBufferedReader(filePath)) {
            try (CSVReader csvReader = new CSVReader(reader)) {

                String[] lineIn;
                String[] lineOut;

                Double dateTimeConvertedToDouble = 0.0;

                int readRecordIndex = 0;

                while ((lineIn = csvReader.readNext()) != null) {

//                    if(readRecordIndex > 0)
//                    {
//
//                        dateTimeConvertedToDouble = (double)convertDateTimeToSysTime(lineIn[4]);
//
//                        lineIn[4] = Double.toString(dateTimeConvertedToDouble);
//
//                    }

                    list.add(lineIn);

                    ++readRecordIndex;

                }
            }
        }

        return list;

    }


    public static List<String[]> readLineByLine2(Path filePath) throws Exception {
        List<String[]> list = new ArrayList<>();
        try (Reader reader = Files.newBufferedReader(filePath)) {
            try (CSVReader csvReader = new CSVReader(reader)) {

                String[] lineIn;
                String[] lineOut = new String[7];

                Double dateTimeConvertedToDouble = 0.0;

                int readRecordIndex = 0;

                while ((lineIn = csvReader.readNext()) != null) {


                    if(readRecordIndex == 0)
                    {
                        lineOut = new String[7];
//                        lineOut = new String[6];

                        lineOut[0] = "latitude";
                        lineOut[1] = "longitude";
                        lineOut[2] = "time";
                        lineOut[3] = "bearing";
                        lineOut[4] = "accuracy";
                        lineOut[5] = "macroMovementLocationChangeFlag";
                        lineOut[6] = "geohash";

                        list.add(lineOut);

                        ++readRecordIndex;

                        continue;

                    }



                    if(readRecordIndex > 0)
                    {

                        lineOut[0] = lineIn[0];
                        lineOut[1] = lineIn[1];
                        lineOut[2] = lineIn[2];

                        lineOut[3] = lineIn[3];
//                        lineOut[3] = lineIn[4];
//                        dateTimeConvertedToDouble = (double)convertDateTimeToSysTime(lineIn[4]);
//                        lineOut[3] = Double.toString(dateTimeConvertedToDouble);

                        lineOut[4] = lineIn[4];
                        lineOut[5] = lineIn[5];

                        lineOut[6] = String.valueOf(GeoHashUtils.encodeAsLong(Double.parseDouble(lineIn[1]),Double.parseDouble(lineIn[2]), 8));

//                        GeoPoint predictedOutputHashString = GeoHashUtils.decode(Long.parseLong(lineOut[5]));
////                        GeoPoint predictedOutputHashString = GeoHashUtils.decode(GeoHashUtils.encodeAsLong(Double.parseDouble(lineIn[1]),Double.parseDouble(lineIn[2]), 8));
//                        double latitude = predictedOutputHashString.getLat();
//                        double longitude = predictedOutputHashString.getLon();
//
//                        Log.info(" - predictedOutputHashString - " + predictedOutputHashString.toString());
//                        Log.info(" - latitude - " + latitude);
//                        Log.info(" - longitude - " + longitude);
//                        Log.info(" - lineIn[1] - " + lineIn[1]);
//                        Log.info(" - lineIn[2] - " + lineIn[2]);
//                        Log.info(" - GeoHashUtils.encodeAsLong(Double.parseDouble(lineIn[1]),Double.parseDouble(lineIn[2]), 8) - " + GeoHashUtils.encodeAsLong(Double.parseDouble(lineIn[1]),Double.parseDouble(lineIn[2]), 8));
//
//                        System.out.println( TAG+" "+"lineOut[5] - "+lineOut[5]);

                    }

                    list.add(lineOut);

                    lineOut = new String[7];

                    ++readRecordIndex;

                }
            }

        }

        return list;

    }

    public static long convertDateTimeToSysTime(String sDateTime) {
        int nYear = Integer.parseInt(sDateTime.substring(0, 4));
        int nMonth = Integer.parseInt(sDateTime.substring(5, 7));
        int nDay = Integer.parseInt(sDateTime.substring(8, 10));
        int nHour = 0, nMinute = 0, nSecond = 0;
        if (sDateTime.length() >= 14) {
            nHour = Integer.parseInt(sDateTime.substring(10, 12));
            nMinute = Integer.parseInt(sDateTime.substring(13, 15));
            nSecond = Integer.parseInt(sDateTime.substring(17));
        }
        Calendar calendar = Calendar.getInstance();
        calendar.set(nYear, nMonth - 1, nDay, nHour, nMinute, nSecond);
        return calendar.getTime().getTime();
    }

    public static void writeAllLines(List<String[]> lines, String path) throws Exception {
        File fileWithDateTimeChangedToDouble = new File(path);
        try (CSVWriter writer = new CSVWriter(new FileWriter(fileWithDateTimeChangedToDouble), ',', CSVWriter.NO_QUOTE_CHARACTER)) {
            writer.writeAll(lines);
        }

    }

    private static XYSeriesCollection createSeries(XYSeriesCollection seriesCollection, INDArray data, int offset, String name) {
        int nRows = (int)data.shape()[2];
        XYSeries series = new XYSeries(name);
        for (int i = 0; i < nRows; i++) {
            series.add(i + offset, data.getDouble(i));
        }

        seriesCollection.addSeries(series);

        return seriesCollection;
    }

    /**
     * Generate an xy plot of the datasets provided.
     */
    private static void plotDataset(XYSeriesCollection c) {

        String title = "Regression example";
        String xAxisLabel = "Timestep";
        String yAxisLabel = "Number of passengers";
        PlotOrientation orientation = PlotOrientation.VERTICAL;
        boolean legend = true;
        boolean tooltips = false;
        boolean urls = false;
        JFreeChart chart = ChartFactory.createXYLineChart(title, xAxisLabel, yAxisLabel, c, orientation, legend, tooltips, urls);

        // get a reference to the plot for further customisation...
        final XYPlot plot = chart.getXYPlot();

        // Auto zoom to fit time series in initial window
        final NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
        rangeAxis.setAutoRange(true);

        JPanel panel = new ChartPanel(chart);

        JFrame f = new JFrame();
        f.add(panel);
        f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        f.pack();
        f.setTitle("Training Data");

        RefineryUtilities.centerFrameOnScreen(f);
        f.setVisible(true);
    }

    /**
     * Licensed to the Apache Software Foundation (ASF) under one or more
     * contributor license agreements.  See the NOTICE file distributed with
     * this work for additional information regarding copyright ownership.
     * The ASF licenses this file to You under the Apache License, Version 2.0
     * (the "License"); you may not use this file except in compliance with
     * the License.  You may obtain a copy of the License at
     *
     *     http://www.apache.org/licenses/LICENSE-2.0
     *
     * Unless required by applicable law or agreed to in writing, software
     * distributed under the License is distributed on an "AS IS" BASIS,
     * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
     * See the License for the specific language governing permissions and
     * limitations under the License.
     */

    /**
     * Utilities for encoding and decoding geohashes. Based on
     * http://en.wikipedia.org/wiki/Geohash.
     */
    // LUCENE MONITOR: monitor against spatial package
    // replaced with native DECODE_MAP
    public static class GeoHashUtils {

        private static final char[] BASE_32 = {'0', '1', '2', '3', '4', '5', '6',
                '7', '8', '9', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'm', 'n',
                'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'};

        public static final int PRECISION = 12;
        private static final int[] BITS = {16, 8, 4, 2, 1};

        private GeoHashUtils() {
        }

        public static String encode(double latitude, double longitude) {
            return encode(latitude, longitude, PRECISION);
        }

        /**
         * Encodes the given latitude and longitude into a geohash
         *
         * @param latitude  Latitude to encode
         * @param longitude Longitude to encode
         * @return Geohash encoding of the longitude and latitude
         */
        public static String encode(double latitude, double longitude, int precision) {
            //        double[] latInterval = {-90.0, 90.0};
            //        double[] lngInterval = {-180.0, 180.0};
            double latInterval0 = -90.0;
            double latInterval1 = 90.0;
            double lngInterval0 = -180.0;
            double lngInterval1 = 180.0;

            final StringBuilder geohash = new StringBuilder();
            boolean isEven = true;

            int bit = 0;
            int ch = 0;

            while (geohash.length() < precision) {
                double mid = 0.0;
                if (isEven) {
                    //                mid = (lngInterval[0] + lngInterval[1]) / 2D;
                    mid = (lngInterval0 + lngInterval1) / 2D;
                    if (longitude > mid) {
                        ch |= BITS[bit];
                        //                    lngInterval[0] = mid;
                        lngInterval0 = mid;
                    } else {
                        //                    lngInterval[1] = mid;
                        lngInterval1 = mid;
                    }
                } else {
                    //                mid = (latInterval[0] + latInterval[1]) / 2D;
                    mid = (latInterval0 + latInterval1) / 2D;
                    if (latitude > mid) {
                        ch |= BITS[bit];
                        //                    latInterval[0] = mid;
                        latInterval0 = mid;
                    } else {
                        //                    latInterval[1] = mid;
                        latInterval1 = mid;
                    }
                }

                isEven = !isEven;

                if (bit < 4) {
                    bit++;
                } else {
                    geohash.append(BASE_32[ch]);
                    bit = 0;
                    ch = 0;
                }
            }

            return geohash.toString();
        }

        private static final char encode(int x, int y) {
            return BASE_32[((x & 1) + ((y & 1) * 2) + ((x & 2) * 2) + ((y & 2) * 4) + ((x & 4) * 4)) % 32];
        }

        /**
         * Calculate all neighbors of a given geohash cell.
         *
         * @param geohash Geohash of the defined cell
         * @return geohashes of all neighbor cells
         */
        public static Collection<? extends CharSequence> neighbors(String geohash) {
            return addNeighbors(geohash, geohash.length(), new ArrayList<CharSequence>(8));
        }

        /**
         * Calculate the geohash of a neighbor of a geohash
         *
         * @param geohash the geohash of a cell
         * @param level   level of the geohash
         * @param dx      delta of the first grid coordinate (must be -1, 0 or +1)
         * @param dy      delta of the second grid coordinate (must be -1, 0 or +1)
         * @return geohash of the defined cell
         */
        private final static String neighbor(String geohash, int level, int dx, int dy) {
            int cell = decode(geohash.charAt(level - 1));

            // Decoding the Geohash bit pattern to determine grid coordinates
            int x0 = cell & 1;  // first bit of x
            int y0 = cell & 2;  // first bit of y
            int x1 = cell & 4;  // second bit of x
            int y1 = cell & 8;  // second bit of y
            int x2 = cell & 16; // third bit of x

            // combine the bitpattern to grid coordinates.
            // note that the semantics of x and y are swapping
            // on each level
            int x = x0 + (x1 / 2) + (x2 / 4);
            int y = (y0 / 2) + (y1 / 4);

            if (level == 1) {
                // Root cells at north (namely "bcfguvyz") or at
                // south (namely "0145hjnp") do not have neighbors
                // in north/south direction
                if ((dy < 0 && y == 0) || (dy > 0 && y == 3)) {
                    return null;
                } else {
                    return Character.toString(encode(x + dx, y + dy));
                }
            } else {
                // define grid coordinates for next level
                final int nx = ((level % 2) == 1) ? (x + dx) : (x + dy);
                final int ny = ((level % 2) == 1) ? (y + dy) : (y + dx);

                // if the defined neighbor has the same parent a the current cell
                // encode the cell directly. Otherwise find the cell next to this
                // cell recursively. Since encoding wraps around within a cell
                // it can be encoded here.
                // xLimit and YLimit must always be respectively 7 and 3
                // since x and y semantics are swapping on each level.
                if (nx >= 0 && nx <= 7 && ny >= 0 && ny <= 3) {
                    return geohash.substring(0, level - 1) + encode(nx, ny);
                } else {
                    String neighbor = neighbor(geohash, level - 1, dx, dy);
                    if(neighbor != null) {
                        return neighbor + encode(nx, ny);
                    } else {
                        return null;
                    }
                }
            }
        }

        /**
         * Add all geohashes of the cells next to a given geohash to a list.
         *
         * @param geohash   Geohash of a specified cell
         * @param neighbors list to add the neighbors to
         * @return the given list
         */
        public static final <E extends Collection<? super String>> E addNeighbors(String geohash, E neighbors) {
            return addNeighbors(geohash, geohash.length(), neighbors);
        }

        /**
         * Add all geohashes of the cells next to a given geohash to a list.
         *
         * @param geohash   Geohash of a specified cell
         * @param length    level of the given geohash
         * @param neighbors list to add the neighbors to
         * @return the given list
         */
        public static final <E extends Collection<? super String>> E addNeighbors(String geohash, int length, E neighbors) {
            String south = neighbor(geohash, length, 0, -1);
            String north = neighbor(geohash, length, 0, +1);
            if (north != null) {
                neighbors.add(neighbor(north, length, -1, 0));
                neighbors.add(north);
                neighbors.add(neighbor(north, length, +1, 0));
            }

            neighbors.add(neighbor(geohash, length, -1, 0));
            neighbors.add(neighbor(geohash, length, +1, 0));

            if (south != null) {
                neighbors.add(neighbor(south, length, -1, 0));
                neighbors.add(south);
                neighbors.add(neighbor(south, length, +1, 0));
            }

            return neighbors;
        }

        private static final int decode(char geo) {
            switch (geo) {
                case '0':
                    return 0;
                case '1':
                    return 1;
                case '2':
                    return 2;
                case '3':
                    return 3;
                case '4':
                    return 4;
                case '5':
                    return 5;
                case '6':
                    return 6;
                case '7':
                    return 7;
                case '8':
                    return 8;
                case '9':
                    return 9;
                case 'b':
                    return 10;
                case 'c':
                    return 11;
                case 'd':
                    return 12;
                case 'e':
                    return 13;
                case 'f':
                    return 14;
                case 'g':
                    return 15;
                case 'h':
                    return 16;
                case 'j':
                    return 17;
                case 'k':
                    return 18;
                case 'm':
                    return 19;
                case 'n':
                    return 20;
                case 'p':
                    return 21;
                case 'q':
                    return 22;
                case 'r':
                    return 23;
                case 's':
                    return 24;
                case 't':
                    return 25;
                case 'u':
                    return 26;
                case 'v':
                    return 27;
                case 'w':
                    return 28;
                case 'x':
                    return 29;
                case 'y':
                    return 30;
                case 'z':
                    return 31;
                default:
                    throw new IllegalArgumentException("the character '" + geo + "' is not a valid geohash character");
            }
        }

        /**
         * Decodes the given geohash
         *
         * @param geohash Geohash to decocde
         * @return {@link GeoPoint} at the center of cell, given by the geohash
         */
        public static GeoPoint decode(String geohash) {
            return decode(geohash, new GeoPoint());
        }

        /**
         * Decodes the given geohash into a latitude and longitude
         *
         * @param geohash Geohash to decocde
         * @return the given {@link GeoPoint} reseted to the center of
         *         cell, given by the geohash
         */
        public static GeoPoint decode(String geohash, GeoPoint ret) {
            double[] interval = decodeCell(geohash);
            return ret.reset((interval[0] + interval[1]) / 2D, (interval[2] + interval[3]) / 2D);
        }

        private static double[] decodeCell(String geohash) {
            double[] interval = {-90.0, 90.0, -180.0, 180.0};
            boolean isEven = true;

            for (int i = 0; i < geohash.length(); i++) {
                final int cd = decode(geohash.charAt(i));

                for (int mask : BITS) {
                    if (isEven) {
                        if ((cd & mask) != 0) {
                            interval[2] = (interval[2] + interval[3]) / 2D;
                        } else {
                            interval[3] = (interval[2] + interval[3]) / 2D;
                        }
                    } else {
                        if ((cd & mask) != 0) {
                            interval[0] = (interval[0] + interval[1]) / 2D;
                        } else {
                            interval[1] = (interval[0] + interval[1]) / 2D;
                        }
                    }
                    isEven = !isEven;
                }
            }
            return interval;
        }

        //========== long-based encodings for geohashes ========================================


        /**
         * Encodes latitude and longitude information into a single long with variable precision.
         * Up to 12 levels of precision are supported which should offer sub-metre resolution.
         *
         * @param latitude
         * @param longitude
         * @param precision The required precision between 1 and 12
         * @return A single long where 4 bits are used for holding the precision and the remaining
         * 60 bits are reserved for 5 bit cell identifiers giving up to 12 layers.
         */
        public static long encodeAsLong(double latitude, double longitude, int precision) {
            if((precision>12)||(precision<1))
            {
                throw new IllegalArgumentException("Illegal precision length of "+precision+
                        ". Long-based geohashes only support precisions between 1 and 12");
            }
            double latInterval0 = -90.0;
            double latInterval1 = 90.0;
            double lngInterval0 = -180.0;
            double lngInterval1 = 180.0;

            long geohash = 0l;
            boolean isEven = true;

            int bit = 0;
            int ch = 0;

            int geohashLength=0;
            while (geohashLength < precision) {
                double mid = 0.0;
                if (isEven) {
                    mid = (lngInterval0 + lngInterval1) / 2D;
                    if (longitude > mid) {
                        ch |= BITS[bit];
                        lngInterval0 = mid;
                    } else {
                        lngInterval1 = mid;
                    }
                } else {
                    mid = (latInterval0 + latInterval1) / 2D;
                    if (latitude > mid) {
                        ch |= BITS[bit];
                        latInterval0 = mid;
                    } else {
                        latInterval1 = mid;
                    }
                }

                isEven = !isEven;

                if (bit < 4) {
                    bit++;
                } else {
                    geohashLength++;
                    geohash|=ch;
                    if(geohashLength<precision){
                        geohash<<=5;
                    }
                    bit = 0;
                    ch = 0;
                }
            }
            geohash<<=4;
            geohash|=precision;
            return geohash;
        }

        /**
         * Formats a geohash held as a long as a more conventional
         * String-based geohash
         * @param geohashAsLong a geohash encoded as a long
         * @return A traditional base32-based String representation of a geohash
         */
        public static String toString(long geohashAsLong)
        {
            int precision = (int) (geohashAsLong&15);
            char[] chars = new char[precision];
            geohashAsLong >>= 4;
            for (int i = precision - 1; i >= 0 ; i--) {
                chars[i] =  BASE_32[(int) (geohashAsLong & 31)];
                geohashAsLong >>= 5;
            }
            return new String(chars);
        }



        public static GeoPoint decode(long geohash) {
            GeoPoint point = new GeoPoint();
            decode(geohash, point);
            return point;
        }

        /**
         * Decodes the given long-format geohash into a latitude and longitude
         *
         * @param geohash long format Geohash to decode
         * @param ret The Geopoint into which the latitude and longitude will be stored
         */
        public static void decode(long geohash, GeoPoint ret) {
            double[] interval = decodeCell(geohash);
            ret.reset((interval[0] + interval[1]) / 2D, (interval[2] + interval[3]) / 2D);

        }

        private static double[] decodeCell(long geohash) {
            double[] interval = {-90.0, 90.0, -180.0, 180.0};
            boolean isEven = true;

            int precision= (int) (geohash&15);
            geohash>>=4;
            int[]cds=new int[precision];
            for (int i = precision-1; i >=0 ; i--) {
                cds[i] = (int) (geohash&31);
                geohash>>=5;
            }

            for (int i = 0; i <cds.length ; i++) {
                final int cd = cds[i];
                for (int mask : BITS) {
                    if (isEven) {
                        if ((cd & mask) != 0) {
                            interval[2] = (interval[2] + interval[3]) / 2D;
                        } else {
                            interval[3] = (interval[2] + interval[3]) / 2D;
                        }
                    } else {
                        if ((cd & mask) != 0) {
                            interval[0] = (interval[0] + interval[1]) / 2D;
                        } else {
                            interval[1] = (interval[0] + interval[1]) / 2D;
                        }
                    }
                    isEven = !isEven;
                }
            }
            return interval;
        }
    }

    /*
     * Licensed to Elasticsearch under one or more contributor
     * license agreements. See the NOTICE file distributed with
     * this work for additional information regarding copyright
     * ownership. Elasticsearch licenses this file to you under
     * the Apache License, Version 2.0 (the "License"); you may
     * not use this file except in compliance with the License.
     * You may obtain a copy of the License at
     *
     *    http://www.apache.org/licenses/LICENSE-2.0
     *
     * Unless required by applicable law or agreed to in writing,
     * software distributed under the License is distributed on an
     * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
     * KIND, either express or implied.  See the License for the
     * specific language governing permissions and limitations
     * under the License.
     */

    /**
     *
     */
    public static class GeoPoint {

        private double lat;
        private double lon;

        public GeoPoint() {
        }

        /**
         * Create a new Geopointform a string. This String must either be a geohash
         * or a lat-lon tuple.
         *
         * @param value String to create the point from
         */
        public GeoPoint(String value) {
            this.resetFromString(value);
        }

        public GeoPoint(double lat, double lon) {
            this.lat = lat;
            this.lon = lon;
        }

        public GeoPoint reset(double lat, double lon) {
            this.lat = lat;
            this.lon = lon;
            return this;
        }

        public GeoPoint resetLat(double lat) {
            this.lat = lat;
            return this;
        }

        public GeoPoint resetLon(double lon) {
            this.lon = lon;
            return this;
        }

        public GeoPoint resetFromString(String value) {
            int comma = value.indexOf(',');
            if (comma != -1) {
                lat = Double.parseDouble(value.substring(0, comma).trim());
                lon = Double.parseDouble(value.substring(comma + 1).trim());
            } else {
                resetFromGeoHash(value);
            }
            return this;
        }

        public GeoPoint resetFromGeoHash(String hash) {
            GeoHashUtils.decode(hash, this);
            return this;
        }

        public final double lat() {
            return this.lat;
        }

        public final double getLat() {
            return this.lat;
        }

        public final double lon() {
            return this.lon;
        }

        public final double getLon() {
            return this.lon;
        }

        public final String geohash() {
            return GeoHashUtils.encode(lat, lon);
        }

        public final String getGeohash() {
            return GeoHashUtils.encode(lat, lon);
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;

            GeoPoint geoPoint = (GeoPoint) o;

            if (Double.compare(geoPoint.lat, lat) != 0) return false;
            if (Double.compare(geoPoint.lon, lon) != 0) return false;

            return true;
        }

        @Override
        public int hashCode() {
            int result;
            long temp;
            temp = lat != +0.0d ? Double.doubleToLongBits(lat) : 0L;
            result = (int) (temp ^ (temp >>> 32));
            temp = lon != +0.0d ? Double.doubleToLongBits(lon) : 0L;
            result = 31 * result + (int) (temp ^ (temp >>> 32));
            return result;
        }

        @Override
        public String toString() {
            return "[" + lat + ", " + lon + "]";
        }

        public static GeoPoint parseFromLatLon(String latLon) {
            GeoPoint point = new GeoPoint();
            point.resetFromString(latLon);
            return point;
        }
    }

}
