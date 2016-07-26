package application;

public class Main {

	public static void main(String[] args) throws Exception {
		String inputFilePath = args[0];
		String outputPath = args[1];
		
		Classification classeur = new Classification(inputFilePath, outputPath);
		
		classeur.startClassification();
	}
	
}
