import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;

public class LR {
	private int vocab_size; 
	private double learn_rate;
	private double regl_coeff; 
	private int max_iter; 
	private int train_size;
	private BufferedReader train; 
	private BufferedReader test;
	//private String trainfile;
	private int[] A;
	private double[][] B;
	
	private static String[] classes={"Activity", "Agent", "Organisation", "other", "Person", 
			"Biomolecule", "CelestialBody", "ChemicalSubstance", "Device",
			"Event", "Location", "Place", "MeanOfTransportation", "Species", 
			"SportsSeason", "TimePeriod", "Work"};
	static private final int NUM_CLASSES=17;
	
	public LR(int vocab_size, double learn_rate, double regl_coeff, int max_iter, int train_size, BufferedReader train, BufferedReader test){
		this.vocab_size=vocab_size;
		this.learn_rate=learn_rate;
		this.regl_coeff=regl_coeff;
		this.max_iter=max_iter;
		this.train_size=train_size;
		this.train=train;
		this.test=test;
		A=new int[vocab_size];
		B=new double[NUM_CLASSES][vocab_size];
		
	}
	


	int k=0;
	
	private void train() throws IOException{
		

		String line;
		int t=1;
		int num_line=0;
		
		
		while ((line=this.train.readLine()) !=null){
			k=k+1;
			double lambda=0.5/(t*t);
			
			String[] tmp=(line.replaceAll("\n", "")).split("\t");
			String[] labels=tmp[1].split(",");
			String[] words=tmp[2].split("\\s");
			for (int i=0; i<NUM_CLASSES; i++){
				
				for (String word: words){
					int id=str2id(word);
					B[i][id]=B[i][id]*Math.pow( (1-2*regl_coeff*lambda), k-A[id] );
				}
				
				double score=0;
				for (String word: words){
					int id=str2id(word);
					score=score+B[i][id];
				}
				double p=sigmoid(score);
				
				int y=0;
				for (String label:labels){
					if (label.equals(classes[i])){
						y=1;
						break;
					}
				}
				for (String word:words){
					int id=str2id(word);
					B[i][id]=B[i][id]+lambda*(y-p);
					A[id]=k;
				}
				
			}

			
			num_line++;
			if (num_line % train_size==0){
				t++;
				for (int i=0; i<NUM_CLASSES; i++){
					for (int j=0; j<vocab_size; j++){
						B[i][j]=B[i][j]*Math.pow( 1-2*regl_coeff*lambda, k-A[j] );
					}
				}
				for (int j=0; j<vocab_size; j++){
					A[j]=k;
				}
				
				if (t>max_iter){
					break;
				}
			}		
		}

		
	}
	

	
	private void test() throws IOException{
		String line;
		int trueCounts=0;
		while ((line=this.test.readLine()) !=null){
			String[] tmp=(line.replaceAll("\n", "")).split("\t");
			String[] labels=tmp[1].split(",");
			String[] words=tmp[2].split("\\s");
			
			double[] p_record=new double[NUM_CLASSES];
			
			StringBuffer str= new StringBuffer();
			
			for (int i=0; i<NUM_CLASSES; i++){
				double score=0;
				for (String word:words){
					int id=str2id(word);
					score=score+ B[i][id];
				}
				double p=sigmoid(score);
				str.append(classes[i]+"\t"+String.valueOf(p)+",");	
				
				
				p_record[i]=p;
				
			}
			
			for (int i=0; i<NUM_CLASSES; i++){
				
				if (Arrays.asList(labels).contains(classes[i])){
					if (p_record[i]>0.5){
						trueCounts++;
					}
				}
				else{
					if (p_record[i]<0.5){
						trueCounts++;
					}
				}
	
			}
			
			
			
			str.deleteCharAt(str.length()-1);
		}
		System.out.println(String.valueOf(trueCounts*1.0/(11188*17)));
		
	}
	
	
	
	
	private static final double OVERFLOW=20;
	protected double sigmoid(double score){
		if (score >OVERFLOW) score=OVERFLOW;
		else if (score < -OVERFLOW) score=-OVERFLOW;
		double exp=Math.exp(score);
		return exp/(1+exp);
	}
	
	protected int str2id(String word){
		int id=word.hashCode() % vocab_size;
		if (id<0) id+=vocab_size;
		return id;
	}
	
	public static void main(String[] args) throws IOException {
		int vocab_size=Integer.parseInt(args[0]);
		double learn_rate=Double.parseDouble(args[1]);
		double regl_coeff=Double.parseDouble(args[2]);
		int max_iter=Integer.parseInt(args[3]);
		int train_size=Integer.parseInt(args[4]);
		
		BufferedReader train=new BufferedReader(new InputStreamReader(System.in));
		BufferedReader test=new BufferedReader(new FileReader(args[5]));
		LR model=new LR( vocab_size,learn_rate,regl_coeff, max_iter, train_size, train, test);
		model.train();
		model.test();
		
	}

}
