import java.io.*;
import java.util.*;
import java.util.Map.Entry;
//import java.util.stream.Collectors;


public class ApproxPageRank {
	private String path;
	private String seed;
	private float alpha;
	private float eps;
	private HashMap <String, Float> p;
	private HashMap <String, Float> r;
	private HashMap <String, List<String>> E;

	
	
	public ApproxPageRank(String path, String seed, float alpha, float eps) throws IOException{
		this.path=path;
		this.alpha=alpha;
		this.seed=seed;
		this.eps=eps;
		this.p=new HashMap <String, Float>();
		this.r=new HashMap <String, Float>();
		
		this.p.put(seed, 0.0f);
		this.r.put(seed, 1.0f);
		
		this.E=new HashMap <String, List<String>>();
		
	}
	

	
	private void getPR() throws IOException{
		String line;

		for (int iter=0; iter<12; iter++){
			BufferedReader file=new BufferedReader (new FileReader(path));

			while ((line =file.readLine()) != null){		
				
				int index=line.indexOf("\t");
				String key=line.substring(0, index);

				if (r.containsKey(key)){
					String[] words=line.split("\t");
					int du=words.length-1;
					if ((r.get(key) / (float) du) < eps){
						continue;
					}
					else{		
						//update edges
						if (!E.containsKey(key)){
							List <String> value= new LinkedList<String>();
							for (int i=1; i<words.length; i++){
								value.add(words[i]);							
							}
							//numE+=words.length-1;
							E.put(key, value);
						}
						
						//update p
						//float score=alpha*r.get(key);
						if (!p.containsKey(key)){
							p.put(key, alpha*r.get(key));

						}
						else{
							p.put(key, p.get(key)+(alpha*r.get(key)));

						}
						
						//update r
						
						float value=((1-alpha)*r.get(key))/2;
						r.put(key,  value);

						
						
						for (int i=1; i<words.length; i++){
							if (!r.containsKey(words[i])){
								r.put(words[i],  value/(float)du);
							}
							else{
								r.put(words[i], r.get(words[i])+value/(float)du);
							}	
						}
					}
				}
				else{
					continue;
				}
			}
			file.close();
			file=null;

		}
		
	}
	
	private void subgraph(){

		List<Entry<String, Float>> list =new LinkedList<Entry<String, Float>>( p.entrySet() );
		Collections.sort( list, new Comparator<Entry<String, Float>>()
		{
			@Override
			public int compare( Entry<String, Float> o1, Entry<String, Float> o2 )
			{
				return ( o2.getValue() ).compareTo( o1.getValue() );
			}
		} );
		
		
		
		HashSet <String> S = new HashSet<String>();
		S.add(seed);
		System.out.println(seed+"\t"+p.get(seed));
		int volume=E.get(seed).size();
		int boundary=E.get(seed).size();
		float conductance=(float) (boundary/volume);

		List <String> l=new LinkedList<String>();
		
		for (Entry<String, Float> entry : list) {
			String key=entry.getKey();
			
			if (key.equals(seed)){
				continue;
			}
			else{
				
				volume=volume+E.get(key).size();
				int count=0;
				for(String e: E.get(key)){
					if (S.contains(e)){
						count++;
					}
				}
				S.add(key);
				l.add(key);
				boundary=boundary+E.get(key).size()-count*2;
				float newconductance= (float)boundary /(float) volume;
				if (conductance > newconductance){
					conductance=newconductance;
					for (String word:l){
						System.out.println(word+"\t"+p.get(word));
					}
					l.clear();
				}
				
			}
		}
		//System.out.println(E.get("Limited_liability_company").size());
	}
	
	



    public static void main(String[] args) throws IOException {
    	String path=args[0];
    	String seed=args[1];
    	Float alpha=Float.valueOf(args[2]);
    	Float eps=Float.valueOf(args[3]);
    	ApproxPageRank pr=new ApproxPageRank(path, seed, alpha, eps);
    	pr.getPR();
    	pr.subgraph();
    	
    }
}