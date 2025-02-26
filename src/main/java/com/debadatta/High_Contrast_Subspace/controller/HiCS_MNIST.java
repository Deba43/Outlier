package com.debadatta.High_Contrast_Subspace.controller;

import java.io.IOException;
import java.util.*;
import org.apache.commons.math3.ml.distance.EuclideanDistance;
import org.apache.commons.math3.stat.inference.KolmogorovSmirnovTest;

public class HiCS_MNIST {
    private static final int NUM_SUBSPACES = 50;
    private static final int NUM_TESTS = 100;
    private static final int NEIGHBORS = 5;
    private static final Random random = new Random();

    public static void main(String[] args) {
        List<double[]> trainDataset = loadMNIST("train");
        int numFeatures = trainDataset.get(0).length;

        List<Set<Integer>> subspaces = generateSubspaces(numFeatures);
        List<Set<Integer>> highContrastSubspaces = selectHighContrastSubspaces(subspaces, trainDataset);
        List<Double> lofScores = computeLOF(highContrastSubspaces, trainDataset);
        displayOutliers(lofScores);
    }

    private static List<double[]> loadMNIST(String type) {
        List<double[]> dataset = new ArrayList<>();
        try {
            CSVDataLoader loader = new CSVDataLoader();
            CSVDataLoader.SimpleDataStore dataStore = loader.loadCSVData(
                    "D:\\High-Contrast-Subspace\\src\\main\\java\\com\\debadatta\\High_Contrast_Subspace\\controller\\diabetes.csv");

            for (int i = 0; i < dataStore.size(); i++) {
                try {
                    double[] dataPoint = filterNumericData(dataStore.get(i).toArray());
                    dataset.add(dataPoint);
                } catch (NumberFormatException e) {
                    System.err.println("Skipping row " + i + " due to non-numeric data: " + e.getMessage());
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        if (dataset.isEmpty()) {
            System.out.println("Error: Dataset is empty.");
        }
        return dataset;
    }

    private static double[] filterNumericData(double[] row) {
        int numColumns = row.length;
        double[] filteredRow = new double[numColumns - 1]; // Exclude the last column
        for (int i = 0; i < numColumns - 1; i++) {
            filteredRow[i] = row[i]; // Copy numeric values
        }
        return filteredRow;
    }

    private static List<Set<Integer>> generateSubspaces(int numFeatures) {
        List<Set<Integer>> subspaces = new ArrayList<>();
        for (int i = 0; i < NUM_SUBSPACES; i++) {
            subspaces.add(selectRandomFeatures(numFeatures));
        }
        return subspaces;
    }

    private static Set<Integer> selectRandomFeatures(int numFeatures) {
        Set<Integer> selectedFeatures = new HashSet<>();
        int targetSize = numFeatures / 2;
        while (selectedFeatures.size() < targetSize) {
            selectedFeatures.add(random.nextInt(numFeatures));
        }
        return selectedFeatures;
    }

    private static List<Set<Integer>> selectHighContrastSubspaces(List<Set<Integer>> subspaces,
            List<double[]> dataset) {
        KolmogorovSmirnovTest ksTest = new KolmogorovSmirnovTest();
        List<Set<Integer>> highContrastSubspaces = new ArrayList<>();
        for (Set<Integer> subspace : subspaces) {
            List<double[]> projectedData = projectData(dataset, subspace);
            double[] sample = generateRandomSample(projectedData);
            double[] flattenedData = flattenData(projectedData);
            double pValue = ksTest.kolmogorovSmirnovTest(flattenedData, sample);
            if (pValue < 0.05) {
                highContrastSubspaces.add(subspace);
            }
        }
        return highContrastSubspaces;
    }

    private static List<double[]> projectData(List<double[]> dataset, Set<Integer> subspace) {
        List<double[]> projected = new ArrayList<>(dataset.size());
        for (double[] data : dataset) {
            projected.add(selectFeatures(data, subspace));
        }
        return projected;
    }

    private static double[] selectFeatures(double[] data, Set<Integer> selectedFeatures) {
        double[] subset = new double[selectedFeatures.size()];
        int index = 0;
        for (int i : selectedFeatures) {
            subset[index++] = data[i];
        }
        return subset;
    }

    private static double[] generateRandomSample(List<double[]> projectedData) {
        int sampleSize = Math.min(1000, projectedData.size());
        double[] sample = new double[sampleSize];
        for (int i = 0; i < sampleSize; i++) {
            double[] randomPoint = projectedData.get(random.nextInt(projectedData.size()));
            sample[i] = randomPoint[random.nextInt(randomPoint.length)];
        }
        return sample;
    }

    private static double[] flattenData(List<double[]> data) {
        int totalSize = data.size() * data.get(0).length;
        double[] flattened = new double[totalSize];
        int index = 0;
        for (double[] vec : data) {
            for (double val : vec) {
                flattened[index++] = val;
            }
        }
        return flattened;
    }

    private static List<Double> computeLOF(List<Set<Integer>> highContrastSubspaces, List<double[]> dataset) {
        List<Double> lofScores = new ArrayList<>(Collections.nCopies(dataset.size(), 0.0));
        int numSubspaces = highContrastSubspaces.size();
        if (numSubspaces == 0)
            return lofScores;

        for (Set<Integer> subspace : highContrastSubspaces) {
            List<double[]> projectedData = projectData(dataset, subspace);
            List<Double> subspaceLOFs = computeLOFForSubspace(projectedData);
            for (int i = 0; i < subspaceLOFs.size(); i++) {
                lofScores.set(i, lofScores.get(i) + subspaceLOFs.get(i));
            }
        }

        // Average the scores
        for (int i = 0; i < lofScores.size(); i++) {
            lofScores.set(i, lofScores.get(i) / numSubspaces);
        }

        return lofScores;
    }

    private static List<Double> computeLOFForSubspace(List<double[]> projectedData) {
        List<Double> lofScores = new ArrayList<>();
        EuclideanDistance distance = new EuclideanDistance();
        int n = projectedData.size();

        for (int i = 0; i < n; i++) {
            double[] point = projectedData.get(i);
            PriorityQueue<Double> distances = new PriorityQueue<>();

            for (int j = 0; j < n; j++) {
                if (i != j) {
                    double dist = distance.compute(point, projectedData.get(j));
                    distances.add(dist);
                }
            }

            double sum = 0;
            for (int k = 0; k < NEIGHBORS; k++) {
                sum += distances.poll();
            }
            lofScores.add(sum / NEIGHBORS);
        }

        return lofScores;
    }

    private static void displayOutliers(List<Double> lofScores) {
        List<Double> sortedScores = new ArrayList<>(lofScores);
        sortedScores.sort(Collections.reverseOrder());
        int topN = Math.min(10, sortedScores.size());
        System.out.println("Top " + topN + " Outliers: " + sortedScores.subList(0, topN));
    }
}