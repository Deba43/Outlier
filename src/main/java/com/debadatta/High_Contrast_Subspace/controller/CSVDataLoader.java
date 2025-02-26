package com.debadatta.High_Contrast_Subspace.controller;

import com.opencsv.CSVReader;
import com.opencsv.exceptions.CsvValidationException;
import de.lmu.ifi.dbs.elki.data.DoubleVector;
import de.lmu.ifi.dbs.elki.database.ids.DBID;
import de.lmu.ifi.dbs.elki.database.ids.DBIDRange;
import de.lmu.ifi.dbs.elki.database.ids.DBIDUtil;
import de.lmu.ifi.dbs.elki.database.ids.DBIDs;
import lombok.Getter;

import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class CSVDataLoader {

    @Getter
    public static class SimpleDataStore {
        private final List<DoubleVector> data;
        private final DBIDRange dbids;

        public SimpleDataStore(List<DoubleVector> data) {
            this.data = data;
            this.dbids = DBIDUtil.generateStaticDBIDRange(data.size());
        }

        public int size() {
            return data.size();
        }

        public DoubleVector get(int index) {
            return data.get(index);
        }

        public DoubleVector get(DBID id) {
            int offset = dbids.getOffset(id);
            return data.get(offset);
        }

        public DBIDs getDBIDs() {
            return dbids;
        }
    }

    public SimpleDataStore loadCSVData(String filePath) throws IOException {
        List<DoubleVector> vectors = new ArrayList<>();

        try (CSVReader reader = new CSVReader(new FileReader(filePath))) {
            // Skip the header row
            reader.readNext();

            String[] nextLine;
            while ((nextLine = reader.readNext()) != null) {
                int numColumns = nextLine.length;
                if (numColumns > 1) { // Ensure there's at least one column to keep
                    double[] values = new double[numColumns - 1]; // Ignore last column
                    for (int i = 0; i < numColumns - 1; i++) {
                        values[i] = Double.parseDouble(nextLine[i]);
                    }
                    vectors.add(new DoubleVector(values));
                }
            }
        } catch (CsvValidationException e) {
            throw new IOException("Invalid CSV format", e);
        }

        return new SimpleDataStore(vectors);
    }

    public static void main(String[] args) {
        try {
            CSVDataLoader loader = new CSVDataLoader();
            SimpleDataStore dataStore = loader.loadCSVData(
                    "D:\\High-Contrast-Subspace\\src\\main\\java\\com\\debadatta\\High_Contrast_Subspace\\controller\\diabetes.csv");

            // Iterate and print the vectors
            dataStore.getData().forEach(System.out::println);

            // Print DBIDs
            DBIDRange range = (DBIDRange) dataStore.getDBIDs();
            for (int i = 0; i < range.size(); i++) {
                DBID id = range.get(i);
                System.out.println("DBID: " + id);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
