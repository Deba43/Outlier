package com.debadatta.High_Contrast_Subspace.controller;

import java.io.FileInputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import org.apache.commons.math3.stat.inference.KolmogorovSmirnovTest;
import weka.core.Instances;
import weka.core.converters.CSVLoader;

public class HiCS {

    private static final int NUM_SUBSPACES = 20; // Number of subspaces to generate
    private static final int SUBSPACE_DIM = 5; // Dimensionality of each subspace
    private static final int NUM_ITERATIONS = 50; // Number of Monte Carlo iterations
    private static final double ALPHA = 0.1; // Size of the test statistic

    public static void main(String[] args) throws Exception {
        // Load MNIST dataset
        CSVLoader loader = new CSVLoader();
        loader.setSource(new FileInputStream(
                "D:\\High-Contrast-Subspace\\src\\main\\java\\com\\debadatta\\High_Contrast_Subspace\\controller\\annthyroid_cleaned.csv"));
        Instances data = loader.getDataSet();
        data.setClassIndex(data.numAttributes() - 1);

        // Filter dataset to only include class 0 and 700 random samples of class 6
        Instances filteredData = filterData(data);

        // Generate random subspaces
        List<int[]> subspaces = generateSubspaces(filteredData.numAttributes() - 1, SUBSPACE_DIM, NUM_SUBSPACES);

        // Calculate contrast for each subspace
        List<Double> contrasts = new ArrayList<>();
        for (int[] subspace : subspaces) {
            double contrast = calculateContrast(filteredData, subspace, NUM_ITERATIONS, ALPHA);
            contrasts.add(contrast);
        }

        // Rank subspaces based on contrast
        List<int[]> rankedSubspaces = rankSubspaces(subspaces, contrasts);

        // Perform outlier ranking in the top subspace
        int[] topSubspace = rankedSubspaces.get(0);
        List<Double> outlierScores = calculateOutlierScores(filteredData, topSubspace);

        // Print top 10 outliers
        printTopOutliers(outlierScores, 10);

        // Compute evaluation metrics
        List<Boolean> actualLabels = new ArrayList<>();
        for (int i = 0; i < filteredData.numInstances(); i++) {
            actualLabels.add(filteredData.instance(i).classValue() == 6);
        }

        int totalPositives = 700; // Number of class 6 instances in filteredData
        int totalNegatives = filteredData.numInstances() - totalPositives;
        int K = Math.min(totalPositives, filteredData.numInstances()); // Ensure K is within bounds

        // Sort instances by outlier scores in descending order
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < outlierScores.size(); i++) {
            indices.add(i);
        }
        indices.sort((i1, i2) -> Double.compare(outlierScores.get(i2), outlierScores.get(i1)));

        // Get top K predicted outliers
        List<Integer> predictedOutliers = indices.subList(0, K);

        // Calculate TP, FP, FN, TN
        int TP = 0, FP = 0;
        for (int idx : predictedOutliers) {
            if (actualLabels.get(idx)) {
                TP++;
            } else {
                FP++;
            }
        }
        int FN = totalPositives - TP;
        int TN = totalNegatives - FP;

        // Compute metrics
        double accuracy = (double) (TP + TN) / (TP + TN + FP + FN);
        double precision = (TP + FP == 0) ? 0 : (double) TP / (TP + FP);
        double recall = (double) TP / totalPositives;
        double falsePositiveRate = (TN + FP == 0) ? 0 : (double) FP / (FP + TN);
        double falseNegativeRate = (FN + TP == 0) ? 0 : (double) FN / (FN + TP);

        // Print metrics
        System.out.println("\nEvaluation Metrics:");
        System.out.println("Confusion Matrix:");
        System.out.println("TP: " + TP + " | FP: " + FP);
        System.out.println("FN: " + FN + " | TN: " + TN);
        System.out.println("Accuracy: " + accuracy);
        System.out.println("Precision: " + precision);
        System.out.println("Recall: " + recall);
        System.out.println("False Positive Rate: " + falsePositiveRate);
        System.out.println("False Negative Rate: " + falseNegativeRate);
    }

    private static Instances filterData(Instances data) {
        Instances filteredData = new Instances(data, 0);
        Random rand = new Random(42); // Fixed seed for reproducibility
        int countClass6 = 0;

        for (int i = 0; i < data.numInstances(); i++) {
            if (data.instance(i).classValue() == 0) {
                filteredData.add(data.instance(i));
            } else if (data.instance(i).classValue() == 6 && countClass6 < 700) {
                filteredData.add(data.instance(i));
                countClass6++;
            }
        }

        return filteredData;
    }

    private static List<int[]> generateSubspaces(int numAttributes, int subspaceDim, int numSubspaces) {
        List<int[]> subspaces = new ArrayList<>();
        Random rand = new Random(42); // Fixed seed for reproducibility

        for (int i = 0; i < numSubspaces; i++) {
            int[] subspace = new int[subspaceDim];
            for (int j = 0; j < subspaceDim; j++) {
                subspace[j] = rand.nextInt(numAttributes);
            }
            subspaces.add(subspace);
        }

        return subspaces;
    }

    private static double calculateContrast(Instances data, int[] subspace, int numIterations, double alpha) {
        KolmogorovSmirnovTest ksTest = new KolmogorovSmirnovTest();
        double totalContrast = 0.0;

        for (int i = 0; i < numIterations; i++) {
            // Randomly select a slice of the subspace
            double[] slice = generateSubspaceSlice(data, subspace, alpha);

            // Calculate marginal and conditional distributions
            double[] marginal = calculateMarginalDistribution(data, subspace[0]);
            double[] conditional = calculateConditionalDistribution(data, subspace, slice);

            // Perform Kolmogorov-Smirnov test
            double contrast = ksTest.kolmogorovSmirnovStatistic(marginal, conditional);
            totalContrast += contrast;
        }

        return totalContrast / numIterations;
    }

    private static double[] generateSubspaceSlice(Instances data, int[] subspace, double alpha) {
        // Randomly select a slice of the subspace
        Random rand = new Random();
        double[] slice = new double[subspace.length - 1];
        for (int i = 1; i < subspace.length; i++) {
            slice[i - 1] = data.instance(rand.nextInt(data.numInstances())).value(subspace[i]);
        }
        return slice;
    }

    private static double[] calculateMarginalDistribution(Instances data, int attribute) {
        if (data.numInstances() < 2) {
            return new double[] { 0.0, 0.0 }; // Return dummy values to avoid errors
        }

        double[] marginal = new double[data.numInstances()];
        for (int i = 0; i < data.numInstances(); i++) {
            marginal[i] = data.instance(i).value(attribute);
        }
        return marginal;
    }

    private static double[] calculateConditionalDistribution(Instances data, int[] subspace, double[] slice) {
        List<Double> conditionalValues = new ArrayList<>();

        for (int i = 0; i < data.numInstances(); i++) {
            boolean match = true;
            for (int j = 1; j < subspace.length; j++) {
                if (Math.abs(data.instance(i).value(subspace[j]) - slice[j - 1]) > 0.1) {
                    match = false;
                    break;
                }
            }
            if (match) {
                conditionalValues.add(data.instance(i).value(subspace[0]));
            }
        }

        // Ensure there are at least two values
        if (conditionalValues.size() < 2) {
            return new double[] { 0.0, 0.0 }; // Return a dummy array to prevent errors
        }

        double[] conditional = new double[conditionalValues.size()];
        for (int i = 0; i < conditionalValues.size(); i++) {
            conditional[i] = conditionalValues.get(i);
        }
        return conditional;
    }

    private static List<int[]> rankSubspaces(List<int[]> subspaces, List<Double> contrasts) {
        List<int[]> rankedSubspaces = new ArrayList<>(subspaces);
        rankedSubspaces.sort(
                (s1, s2) -> Double.compare(contrasts.get(subspaces.indexOf(s2)), contrasts.get(subspaces.indexOf(s1))));
        return rankedSubspaces;
    }

    private static List<Double> calculateOutlierScores(Instances data, int[] subspace) {
        List<Double> outlierScores = new ArrayList<>();

        // Compute outlier scores based on density
        for (int i = 0; i < data.numInstances(); i++) {
            double density = calculateDensity(data, subspace, i);
            outlierScores.add(Math.log(1 + (1 / (density + 1e-10))));
        }

        // Min-Max Normalization to scale scores between 0 and 1
        double maxScore = Collections.max(outlierScores);
        double minScore = Collections.min(outlierScores);

        if (maxScore != minScore) { // Avoid division by zero
            for (int i = 0; i < outlierScores.size(); i++) {
                outlierScores.set(i, (outlierScores.get(i) - minScore) / (maxScore - minScore));
            }
        }

        return outlierScores;
    }

    private static double calculateDensity(Instances data, int[] subspace, int index) {
        double density = 0.0;
        for (int i = 0; i < data.numInstances(); i++) {
            if (i != index) {
                double distance = calculateDistance(data.instance(index), data.instance(i), subspace);
                density += Math.exp(-0.01 * distance * distance);

            }
        }
        return density / (data.numInstances() - 1);
    }

    private static double calculateDistance(weka.core.Instance inst1, weka.core.Instance inst2, int[] subspace) {
        double distance = 0.0;
        for (int attr : subspace) {
            distance += Math.pow(inst1.value(attr) - inst2.value(attr), 2);
        }
        return Math.sqrt(distance);
    }

    private static void printTopOutliers(List<Double> outlierScores, int topN) {
        List<Integer> outlierIndices = new ArrayList<>();
        for (int i = 0; i < outlierScores.size(); i++) {
            outlierIndices.add(i);
        }
        outlierIndices.sort((i1, i2) -> Double.compare(outlierScores.get(i2), outlierScores.get(i1)));

        System.out.println("Top " + topN + " Outliers:");
        for (int i = 0; i < topN; i++) {
            System.out.println(
                    "Index: " + outlierIndices.get(i) + ", Score: " + outlierScores.get(outlierIndices.get(i)));
        }
    }
}