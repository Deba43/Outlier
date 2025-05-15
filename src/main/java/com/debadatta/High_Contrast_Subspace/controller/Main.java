package com.debadatta.High_Contrast_Subspace.controller;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;
import org.knowm.xchart.*;
import org.knowm.xchart.style.markers.SeriesMarkers;

public class Main {

    static class DataInstance {
        double[] features;
        int label;
        double lofScore;

        DataInstance(double[] features, int label) {
            this.features = features;
            this.label = label;
        }
    }

    static class EvaluationMetrics {
        int truePositives;
        int trueNegatives;
        int falsePositives;
        int falseNegatives;
        double[] rocCurve;
        double auc;

        @Override
        public String toString() {
            return String.format(
                    "Confusion Matrix:\n" +
                            "TP: %d, FP: %d\n" +
                            "FN: %d, TN: %d\n" +
                            "AUC: %.4f",
                    truePositives, falsePositives,
                    falseNegatives, trueNegatives,
                    auc);
        }
    }

    static List<DataInstance> loadCSV(String filename) throws IOException {
        List<DataInstance> data = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line;

            // Skip header
            if (br.readLine() == null) {
                System.err.println("Warning: Empty file");
                return data;
            }

            int lineNum = 1;
            while ((line = br.readLine()) != null) {
                lineNum++;
                line = line.trim();
                if (line.isEmpty())
                    continue;

                String[] parts = line.split(",");
                if (parts.length < 2) {
                    System.err.println("Skipping line " + lineNum + ": Not enough columns");
                    continue;
                }

                try {
                    double[] features = new double[parts.length - 1];
                    boolean valid = true;

                    // Validate features
                    for (int i = 0; i < parts.length - 1; i++) {
                        try {
                            features[i] = Double.parseDouble(parts[i].trim());
                        } catch (NumberFormatException e) {
                            System.err.println("Line " + lineNum + ", column " + (i + 1) + ": Invalid value '"
                                    + parts[i] + "', using 0.0");
                            features[i] = 0.0; // Default value for invalid entries
                            valid = false;
                        }
                    }

                    // Validate label
                    int label = 0;
                    try {
                        label = Integer.parseInt(parts[parts.length - 1].trim());
                    } catch (NumberFormatException e) {
                        System.err.println(
                                "Line " + lineNum + ": Invalid label '" + parts[parts.length - 1] + "', using 0");
                        valid = false;
                    }

                    if (valid || data.isEmpty()) { // Allow some invalid entries if needed
                        data.add(new DataInstance(features, label));
                    }
                } catch (Exception e) {
                    System.err.println("Skipping line " + lineNum + ": " + e.getMessage());
                }
            }
        }
        return data;
    }

    static void normalize(List<DataInstance> data) {
        if (data.isEmpty())
            return;

        int dim = data.get(0).features.length;
        double[] mean = new double[dim];
        double[] std = new double[dim];

        for (int i = 0; i < dim; i++) {
            final int index = i;
            mean[i] = data.stream().mapToDouble(d -> d.features[index]).average().orElse(0);
            std[i] = Math.sqrt(
                    data.stream().mapToDouble(d -> Math.pow(d.features[index] - mean[index], 2)).average().orElse(1));
        }

        for (DataInstance d : data) {
            for (int i = 0; i < dim; i++) {
                d.features[i] = (d.features[i] - mean[i]) / std[i];
            }
        }
    }

    static double ksStatistic(List<Double> all, List<Double> slice) {
        Collections.sort(all);
        Collections.sort(slice);
        double d = 0;
        int n1 = all.size();
        int n2 = slice.size();
        for (int i = 0; i < all.size(); i++) {
            double v = all.get(i);
            double cdf1 = (i + 1.0) / n1;
            double cdf2 = (double) slice.stream().filter(s -> s <= v).count() / n2;
            d = Math.max(d, Math.abs(cdf1 - cdf2));
        }
        return d;
    }

    static double computeContrast(List<DataInstance> data, int[] subspace, int numTests, double alpha) {
        Random rand = new Random();
        int dim = subspace.length;
        double totalKS = 0;

        for (int t = 0; t < numTests; t++) {
            int targetIndex = subspace[rand.nextInt(dim)];
            List<DataInstance> slice = new ArrayList<>();

            // Pick random ranges on d-1 attributes
            for (DataInstance d : data) {
                boolean match = true;
                for (int i = 0; i < dim; i++) {
                    if (subspace[i] == targetIndex)
                        continue;
                    double val = d.features[subspace[i]];
                    double min = -1.0, max = 1.0; // normalized range
                    if (val < min || val > max) {
                        match = false;
                        break;
                    }
                }
                if (match)
                    slice.add(d);
            }

            if (slice.size() < 10)
                continue;

            List<Double> full = data.stream().map(d -> d.features[targetIndex]).collect(Collectors.toList());
            List<Double> part = slice.stream().map(d -> d.features[targetIndex]).collect(Collectors.toList());

            totalKS += ksStatistic(full, part);
        }

        return totalKS / numTests;
    }

    static List<int[]> getTopKSubspaces(List<DataInstance> data, int subDim, int topK, int numTests, double alpha) {
        if (data.isEmpty())
            return Collections.emptyList();

        int dim = data.get(0).features.length;
        List<int[]> candidates = new ArrayList<>();

        // generate combinations
        int[] indices = new int[subDim];
        for (int i = 0; i < subDim; i++)
            indices[i] = i;

        while (indices[0] <= dim - subDim) {
            candidates.add(Arrays.copyOf(indices, subDim));
            int i = subDim - 1;
            indices[i]++;
            while (i > 0 && indices[i] >= dim - (subDim - 1 - i)) {
                i--;
                indices[i]++;
                for (int j = i + 1; j < subDim; j++) {
                    indices[j] = indices[j - 1] + 1;
                }
            }
        }

        List<Map.Entry<int[], Double>> scored = new ArrayList<>();
        for (int[] sub : candidates) {
            double contrast = computeContrast(data, sub, numTests, alpha);
            scored.add(new AbstractMap.SimpleEntry<>(sub, contrast));
        }

        scored.sort((a, b) -> Double.compare(b.getValue(), a.getValue()));
        return scored.subList(0, topK).stream().map(Map.Entry::getKey).collect(Collectors.toList());
    }

    static class Neighbor implements Comparable<Neighbor> {
        int index;
        double distance;

        Neighbor(int index, double distance) {
            this.index = index;
            this.distance = distance;
        }

        @Override
        public int compareTo(Neighbor o) {
            return Double.compare(this.distance, o.distance);
        }
    }

    static double euclidean(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            double d = a[i] - b[i];
            sum += d * d;
        }
        return Math.sqrt(sum);
    }

    // [Previous methods: loadCSV, normalize, ksStatistic, computeContrast,
    // getTopKSubspaces, Neighbor, euclidean remain exactly the same]

    static double[] computeLOF(List<DataInstance> data, int k) {
        int n = data.size();
        if (n <= k) {
            System.err.println("Not enough data points for LOF calculation (need > k)");
            return new double[n];
        }

        double[][] distances = new double[n][n];
        List<List<Neighbor>> neighbors = new ArrayList<>();

        // Step 1: Compute pairwise distances
        for (int i = 0; i < n; i++) {
            List<Neighbor> neighList = new ArrayList<>();
            for (int j = 0; j < n; j++) {
                if (i == j)
                    continue;
                double dist = euclidean(data.get(i).features, data.get(j).features);
                distances[i][j] = dist;
                neighList.add(new Neighbor(j, dist));
            }
            Collections.sort(neighList);
            neighbors.add(neighList);
        }

        // Step 2: Compute k-distance and reachability distances
        double[][] reachDist = new double[n][n];
        double[] lrd = new double[n];

        for (int i = 0; i < n; i++) {
            if (neighbors.get(i).size() < k) {
                lrd[i] = 0;
                continue;
            }

            List<Neighbor> neigh = neighbors.get(i).subList(0, k);
            double reachSum = 0;

            for (Neighbor nb : neigh) {
                int j = nb.index;
                double kdist = neighbors.get(j).get(k - 1).distance;
                reachDist[i][j] = Math.max(distances[i][j], kdist);
                reachSum += reachDist[i][j];
            }

            lrd[i] = k / reachSum;
        }

        // Step 3: Compute LOF
        double[] lof = new double[n];
        for (int i = 0; i < n; i++) {
            if (neighbors.get(i).size() < k) {
                lof[i] = 0;
                continue;
            }

            List<Neighbor> neigh = neighbors.get(i).subList(0, k);
            double sum = 0;
            for (Neighbor nb : neigh) {
                int j = nb.index;
                sum += lrd[j] / lrd[i];
            }
            lof[i] = sum / k;
            data.get(i).lofScore = lof[i]; // Store LOF score in the instance
        }

        return lof;
    }

    static EvaluationMetrics evaluatePerformance(List<DataInstance> data, double threshold) {
        EvaluationMetrics metrics = new EvaluationMetrics();

        // Calculate confusion matrix
        for (DataInstance instance : data) {
            boolean isAnomaly = instance.lofScore > threshold;
            boolean isActualAnomaly = instance.label == 1; // Assuming 1=anomaly, 0=normal

            if (isActualAnomaly && isAnomaly)
                metrics.truePositives++;
            else if (!isActualAnomaly && !isAnomaly)
                metrics.trueNegatives++;
            else if (!isActualAnomaly && isAnomaly)
                metrics.falsePositives++;
            else if (isActualAnomaly && !isAnomaly)
                metrics.falseNegatives++;
        }

        // Calculate ROC curve and AUC
        int numSteps = 100;
        metrics.rocCurve = new double[numSteps * 2]; // [fpr, tpr] pairs

        double[] scores = data.stream().mapToDouble(d -> d.lofScore).toArray();
        int[] labels = data.stream().mapToInt(d -> d.label).toArray();

        double minScore = Arrays.stream(scores).min().orElse(0);
        double maxScore = Arrays.stream(scores).max().orElse(1);

        for (int i = 0; i < numSteps; i++) {
            double currentThreshold = minScore + (maxScore - minScore) * i / numSteps;

            int tp = 0, fp = 0, tn = 0, fn = 0;
            for (int j = 0; j < data.size(); j++) {
                boolean isAnomaly = scores[j] > currentThreshold;
                boolean isActualAnomaly = labels[j] == 1;

                if (isActualAnomaly && isAnomaly)
                    tp++;
                else if (!isActualAnomaly && !isAnomaly)
                    tn++;
                else if (!isActualAnomaly && isAnomaly)
                    fp++;
                else if (isActualAnomaly && !isAnomaly)
                    fn++;
            }

            double tpr = (tp + fn) == 0 ? 0 : (double) tp / (tp + fn);
            double fpr = (fp + tn) == 0 ? 0 : (double) fp / (fp + tn);

            metrics.rocCurve[i * 2] = fpr;
            metrics.rocCurve[i * 2 + 1] = tpr;
        }

        // Calculate AUC using trapezoidal rule
        metrics.auc = 0;
        for (int i = 1; i < numSteps; i++) {
            double x1 = metrics.rocCurve[(i - 1) * 2];
            double y1 = metrics.rocCurve[(i - 1) * 2 + 1];
            double x2 = metrics.rocCurve[i * 2];
            double y2 = metrics.rocCurve[i * 2 + 1];

            metrics.auc += (x2 - x1) * (y1 + y2) / 2;
        }

        return metrics;
    }

    static void plotROCCurve(EvaluationMetrics metrics, String title) {
        // Extract FPR and TPR from rocCurve array
        List<Double> fprList = new ArrayList<>();
        List<Double> tprList = new ArrayList<>();

        for (int i = 0; i < metrics.rocCurve.length / 2; i++) {
            fprList.add(metrics.rocCurve[i * 2]);
            tprList.add(metrics.rocCurve[i * 2 + 1]);
        }

        // Create Chart
        XYChart chart = new XYChartBuilder()
                .width(800)
                .height(600)
                .title(title)
                .xAxisTitle("False Positive Rate")
                .yAxisTitle("True Positive Rate")
                .build();

        chart.getStyler().setLegendVisible(false);
        chart.getStyler().setMarkerSize(6);

        // Add ROC Curve
        XYSeries series = chart.addSeries("ROC Curve", fprList, tprList);
        series.setMarker(SeriesMarkers.CIRCLE);

        // Add diagonal line (random classifier)
        chart.addSeries("Random Classifier", Arrays.asList(0.0, 1.0), Arrays.asList(0.0, 1.0))
                .setMarker(SeriesMarkers.NONE);

        // Show the chart
        new SwingWrapper<>(chart).displayChart();
    }

    public static void main(String[] args) throws IOException {
        String filePath = "D:\\\\High-Contrast-Subspace\\\\src\\\\main\\\\java\\\\com\\\\debadatta\\\\High_Contrast_Subspace\\\\controller\\\\ann-test.csv";
        List<DataInstance> data = loadCSV(filePath);

        if (data.isEmpty()) {
            System.err.println("Error: No data loaded");
            return;
        }
        System.out.println("Loaded " + data.size() + " instances");

        normalize(data);

        // Parameters
        int subspaceDim = 3;
        int topK = 50;
        int numTests = 40;
        double alpha = 0.05;

        // Get top-k high contrast subspaces
        List<int[]> topSubspaces = getTopKSubspaces(data, subspaceDim, topK, numTests, alpha);
        System.out.println("Top subspaces: " + topSubspaces);

        // Compute LOF
        int kNeighbors = 50;
        double[] lofScores = computeLOF(data, kNeighbors);

        // Set LOF scores in data instances
        for (int i = 0; i < data.size(); i++) {
            data.get(i).lofScore = lofScores[i];
        }

        // Evaluate performance
        double threshold = 1.7; // You may need to adjust this based on your data
        EvaluationMetrics metrics = evaluatePerformance(data, threshold);

        System.out.println("\nPerformance Metrics:");
        System.out.println(metrics);

        // Plot ROC curve
        plotROCCurve(metrics, "ROC Curve for Anomaly Detection");
    }
}