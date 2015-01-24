package weka.clusterers;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.Vector;

import weka.clusterers.forOPTICSAndDBScan.DataObjects.DataObject;
import weka.clusterers.forOPTICSAndDBScan.Databases.Database;
import weka.clusterers.forOPTICSAndDBScan.Databases.SequentialDatabase;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.core.Capabilities.Capability;

public class dbMedoids extends RandomizableClusterer 
implements NumberOfClustersRequestable, WeightedInstancesHandler{
	
	/**
	 * 
	 */
	//Kdd elpased time 22K instances : 2471.83s
	//App parameters
	private static final long serialVersionUID = 5906266235776554886L;
	private double epsilon = 0.9;
	private int minPoints = 6;
	private int numClusters = 2;
	private int maxIterations = 300;
	//attributes needed
	//store results
	private double elapsedTime;
	private int clusterID;
	private int numDensityGeneratedClusters;
	private int processed_InstanceID;
	
	//needed for processing
	private Database database;
	private ArrayList<Integer> medoidList;
	private ArrayList<Integer> DensityMedoidList;
	private ArrayList<Double> m_DistanceErrors;
	private ArrayList<Double> radiuses;
	private ArrayList<Integer> deletedClustersList;
	private int tmpNumClusters;
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();
        result.enable(Capability.NO_CLASS);

        // attributes
        result.enable(Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capability.DATE_ATTRIBUTES);
        //result.enable(Capability.MISSING_VALUES);

        return result;
     }
    
	@Override
	public void buildClusterer(Instances data) throws Exception {
		getCapabilities().testWithFail(data);
		//Preparations and initializations
		numDensityGeneratedClusters = 0;
        clusterID = 0;
		processed_InstanceID=0;
		this.medoidList = new ArrayList<Integer>(this.numClusters);
		
		long time_1 = System.currentTimeMillis();
		 database = databaseForName(data);
	     for (int i = 0; i < database.getInstances().numInstances(); i++) {
	         DataObject dataObject = dataObjectForName(
	                    database.getInstances().instance(i),
	                    Integer.toString(i),
	                    database);
	            database.insert(dataObject);
	     }
	     database.setMinMaxValues();
	     
	     Iterator<DataObject> iterator = database.dataObjectIterator();
	     
	     //Density part of the algorithm
	     while(iterator.hasNext()){
	    	 DataObject point=iterator.next();
	    	 
	    	 if(point.getClusterLabel() == DataObject.UNCLASSIFIED){
	    		 List<DataObject> neighborhood=database.epsilonRangeQuery(this.epsilon, point);
	    		 if(neighborhood.size() < this.minPoints){
	    			 point.setClusterLabel(DataObject.NOISE);
	    			 
	    		 }else{
	    			 addToACluster(neighborhood, point);
	    			 recurse(neighborhood);
	    			 
	    			 clusterID++;
	    			 numDensityGeneratedClusters++;
	    		 }
	    	 }
	     }
	     
	     
	     //Medoids part
	     
	     this.DensityMedoidList = new ArrayList<Integer>(this.numDensityGeneratedClusters);
	     this.m_DistanceErrors = new ArrayList<Double>(this.numDensityGeneratedClusters);
	     this.deletedClustersList = new ArrayList<Integer>(this.numDensityGeneratedClusters);
	     //get medoid of each DB cluster
	     for(int i=0;i<this.numDensityGeneratedClusters;i++){
	    	//initialization and preparation of variables
	    	 m_DistanceErrors.add(i,Double.MAX_VALUE);
	    	 this.DensityMedoidList.add(i,0);
	    	 //get medoids
	    	 updateMedoid(i,getClusterByIndex(i), this.DensityMedoidList);
	    	 //System.out.println(this.DensityMedoidList.get(i));
	     }
	     //assign each instance to the closest medoid
	     for(int i=0;i<database.size();i++){
			DataObject tmpPoint=database.getDataObject(Integer.toString(i));
			assignToClosest(this.DensityMedoidList, tmpPoint);
	     }
	     
	     
	     radiuses = new ArrayList<Double>(this.numDensityGeneratedClusters);
	     
	     
	     this.tmpNumClusters=this.numDensityGeneratedClusters;
	     
		    while (this.tmpNumClusters != this.numClusters){
		    	
		    	 if(this.tmpNumClusters > this.numClusters){
		    		 // when Number of Density Generated Clusters > Number of wanted Clusters, join until we reach numClusters wanted
		    		GetAndAsigneToMedoids(this.DensityMedoidList);
		    		joinClusters();
		 	     }else if(this.tmpNumClusters < this.numClusters){
		 	    	GetAndAsigneToMedoids(this.DensityMedoidList);
		 	    	split(); 
		 	     } 
			}
	     
		 //one last iteration of k-Medoids
		    boolean changed=true;
		    int iter=0;
		    while(changed && iter<=this.maxIterations){
		    	GetAndAsigneToMedoids(this.DensityMedoidList);
		    	iter++;
		    }
		 //end   
		   
	     int t=0;
	     if(this.tmpNumClusters<this.numDensityGeneratedClusters){
		     for(int i=0;i<this.numDensityGeneratedClusters;i++){
		    	 if(!this.deletedClustersList.contains(i)){
		    		 for(Integer cluster:getClusterByIndex(i)){
		    			 database.getDataObject(Integer.toString(cluster)).setClusterLabel(t);
		    		 }
		    		 t++;
		    	 }
		     }
		 }
	     long time_2 = System.currentTimeMillis();
	     elapsedTime = (double) (time_2 - time_1) / 1000.0;
	}
	
	public void split(){
		double minDensity=Double.MAX_VALUE;
		double tmpDensity;
		int ClusterToSplitIndex = 0;
		for(int i=0;i<this.DensityMedoidList.size();i++){
			tmpDensity=getClusterByIndex(i).size() / (2*clusterRadius(i));
			if(tmpDensity<minDensity){
				minDensity=tmpDensity;
				ClusterToSplitIndex=i;
			}
		}
		
		ArrayList<Integer> ClusterToSplit=getClusterByIndex(ClusterToSplitIndex);
		Random rand = new Random();
		int  n = rand.nextInt(ClusterToSplit.size()) + 1;
		this.DensityMedoidList.add(ClusterToSplit.get(n));
		for (int i = 0; i < ClusterToSplit.size(); i++) {
			assignToClosest(this.DensityMedoidList, database.getDataObject(Integer.toString(ClusterToSplit.get(i))));
		}
		
		updateMedoid(ClusterToSplitIndex, ClusterToSplit, this.DensityMedoidList);
		updateMedoid(this.DensityMedoidList.size()-1, ClusterToSplit, this.DensityMedoidList);
		this.tmpNumClusters++;
	}
	
	public boolean GetAndAsigneToMedoids(ArrayList<Integer> medoids){
		boolean changed=true;
		for(int i=0;i<medoids.size();i++){
	    	 //get medoids
			if(!this.deletedClustersList.contains(i)){
				changed= changed || updateMedoid(i,getClusterByIndex(i), medoids);
			}
	    	 
	     }
	     //assign each instance to the closest medoid
	     for(int i=0;i<database.size();i++){
	    	 if(!this.deletedClustersList.contains(i)){
	    		 DataObject tmpPoint=database.getDataObject(Integer.toString(i));
	 			 assignToClosest(medoids, tmpPoint); 
	    	 }
	     }
	     return changed;
	}
	
	public void joinClusters(){
		double minDisim=Double.MAX_VALUE;
		int indexCluster1=0;
		int indexCluster2=0;
		
		
		for(int i=0;i<this.DensityMedoidList.size();i++){
			if(!this.deletedClustersList.contains(i)){
				
   		 		radiuses.add(i,clusterRadius(i));
			}
   	 	}
		for(int i=0;i<this.DensityMedoidList.size();i++){
			for(int j=i+1;j<this.DensityMedoidList.size();j++){
				if(!this.deletedClustersList.contains(i) && !this.deletedClustersList.contains(j)){
					double tmpDisim=clusterDisimilarity(i, j);
					if(tmpDisim<minDisim){
						minDisim=tmpDisim;
						indexCluster1=i;
						indexCluster2=j;
					}
				}
			}
		}
		ArrayList<Integer> tmpCluster=getClusterByIndex(Math.max(indexCluster1,indexCluster2));
		for(int i=0;i<tmpCluster.size();i++){
			this.database.getDataObject(Integer.toString(tmpCluster.get(i))).setClusterLabel(Math.min(indexCluster1,indexCluster2));
		}
		
		this.tmpNumClusters--;
		this.deletedClustersList.add(Math.max(indexCluster1,indexCluster2));		
	}
	public double clusterDisimilarity(int indexCluster_i,int indexCluster_j){
		double disimilarity=0;
		DataObject clusterMedoid_i=database.getDataObject(Integer.toString(this.DensityMedoidList.get(indexCluster_i)));
		DataObject clusterMedoid_j=database.getDataObject(Integer.toString(this.DensityMedoidList.get(indexCluster_j)));
		disimilarity=(clusterMedoid_i.distance(clusterMedoid_j))/(this.radiuses.get(indexCluster_i)+this.radiuses.get(indexCluster_j));
		return disimilarity;
	}
	public double clusterRadius(int indexCluster){
		Double radius = 0.0;
		Double tmpDist=0.0;
		ArrayList<Integer> cluster=getClusterByIndex(indexCluster);
		DataObject clusterMedoid=database.getDataObject(Integer.toString(this.DensityMedoidList.get(indexCluster)));
		for(int i=0;i<cluster.size();i++){
			tmpDist=database.getDataObject(Integer.toString(cluster.get(i))).distance(clusterMedoid);
			if(tmpDist>radius){
				radius=tmpDist;
			}
		}
		return radius;
	}
	
	public void assignToClosest(ArrayList<Integer> medoids,DataObject instance){
		double minDist=database.getDataObject(Integer.toString(medoids.get(0))).distance(instance); 
		double tmpDist;
		Integer closest=0;
		for(int i=1;i<medoids.size();i++){
			if(!this.deletedClustersList.contains(i)){
				tmpDist=database.getDataObject(Integer.toString(medoids.get(i))).distance(instance); 
				if(tmpDist<minDist){
					minDist=tmpDist;
					closest=i;
				}
			}
			
		}
		instance.setClusterLabel(closest);
	}
	
	public ArrayList<Integer> getClusterByIndex(int index){
		ArrayList<Integer> tmpList=new ArrayList<Integer>();
		for(int i=0;i<database.size();i++){
			DataObject tmpPoint=database.getDataObject(Integer.toString(i));
			if(tmpPoint.getClusterLabel()==index){
				tmpList.add(i);
			}
		}
		return tmpList;
	}

	
	private boolean updateMedoid(int clusterIndex, ArrayList<Integer> clusterMember,ArrayList<Integer> medoids) {
		double BestCost = m_DistanceErrors.get(clusterIndex);
		int NewMedoid = medoids.get(clusterIndex);
		int ClusterSize = clusterMember.size();
		for (int i=0; i<ClusterSize; i++) {
			double CurrentCost = 0;
			int CurrentMedoid = clusterMember.get(i);
			for (Integer x : clusterMember) {
				CurrentCost +=database.getDataObject(Integer.toString(CurrentMedoid)).distance(database.getDataObject(Integer.toString(x))); 
			}
			if (CurrentCost < BestCost) {
				NewMedoid = CurrentMedoid;
				BestCost = CurrentCost;
			}
		}
		if (NewMedoid == medoids.get(clusterIndex)) {
			return false;	//No change
		}else{
			medoids.set(clusterIndex, NewMedoid);
			m_DistanceErrors.add(clusterIndex,BestCost);
			return true;
		}
	}
	
	public void addToACluster(List<DataObject> neighborhood,DataObject point){
        for (int i = 0; i < neighborhood.size(); i++) {
            DataObject neighborhoodPoint = (DataObject) neighborhood.get(i);
            
            neighborhoodPoint.setClusterLabel(clusterID);
            if (neighborhoodPoint.equals(point)) {
            	neighborhood.remove(i);
                i--;
            }
        }
	}
	/*public void recurse(List<DataObject> neighborhood) throws Exception{
		
		for (int j = 0; j < neighborhood.size(); j++) {
            DataObject neighborhoodPoint = (DataObject) neighborhood.get(j);
            List<DataObject> neighborhoodPoint_Neighbourhood = database.epsilonRangeQuery(getEpsilon(), neighborhoodPoint);

            // seedListDataObject is coreObject 
            if (neighborhoodPoint_Neighbourhood.size() >= getMinPoints()) {
                if (neighborhoodPoint.getClusterLabel() == DataObject.UNCLASSIFIED || neighborhoodPoint.getClusterLabel() == DataObject.NOISE) {
                    neighborhoodPoint.setClusterLabel(clusterID);
                }
            }
        }
		
	}*/
	
	public void recurse(List<DataObject> neighborhood) throws Exception{
		
		for (int j = 0; j < neighborhood.size(); j++) {
            DataObject neighborhoodPoint = (DataObject) neighborhood.get(j);
            List<DataObject> neighborhoodPoint_Neighbourhood = database.epsilonRangeQuery(getEpsilon(), neighborhoodPoint);

            /** seedListDataObject is coreObject */
            if (neighborhoodPoint_Neighbourhood.size() >= getMinPoints()) {
                for (int i = 0; i < neighborhoodPoint_Neighbourhood.size(); i++) {
                    DataObject p = (DataObject) neighborhoodPoint_Neighbourhood.get(i);
                    if (p.getClusterLabel() == DataObject.UNCLASSIFIED || p.getClusterLabel() == DataObject.NOISE) {
                        if (p.getClusterLabel() == DataObject.UNCLASSIFIED) {
                        	neighborhood.add(p);
                        }
                        p.setClusterLabel(clusterID);
                    }
                }
            }
            neighborhood.remove(j);
            j--;
            
        }
		
	}
	@Override
	public int numberOfClusters() throws Exception {
		// TODO Auto-generated method stub
		return this.tmpNumClusters;
	}
	
	public String toString() {
        StringBuffer stringBuffer = new StringBuffer();
        stringBuffer.append("DBSCAN clustering results\n" +
                "========================================================================================\n\n");
        stringBuffer.append("Clustered DataObjects: " + database.size() + "\n");
        stringBuffer.append("Number of attributes: " + database.getInstances().numAttributes() + "\n");
        try {
			stringBuffer.append("Epsilon: " + getEpsilon() + "; minPoints: " + getMinPoints() + "\n");
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
        stringBuffer.append("Number of generated clusters: " + tmpNumClusters + "\n");
        DecimalFormat decimalFormat = new DecimalFormat(".##");
        stringBuffer.append("Elapsed time: " + decimalFormat.format(elapsedTime) + "\n\n");
/*
        for (int i = 0; i < database.size(); i++) {
            DataObject dataObject = database.getDataObject(Integer.toString(i));
            stringBuffer.append("(" + Utils.doubleToString(Double.parseDouble(dataObject.getKey()),
                    (Integer.toString(database.size()).length()), 0) + ".) "
                    + Utils.padRight(dataObject.toString(), 69) + "  -->  " +
                    ((dataObject.getClusterLabel() == DataObject.NOISE) ?
                    "NOISE\n" : dataObject.getClusterLabel() + "\n"));
        }*/
        return stringBuffer.toString() + "\n";
    }
	
	public int clusterInstance(Instance instance) throws Exception {
        if (processed_InstanceID >= database.size()) processed_InstanceID = 0;
        int cnum = (database.getDataObject(Integer.toString(processed_InstanceID++))).getClusterLabel();
        if (cnum == DataObject.NOISE)
            throw new Exception();
        else
            return cnum;
    }
	
	public DataObject dataObjectForName(Instance instance, String key, Database database) {
        Object o = null;

        Constructor co = null;
        try {
            co = (Class.forName("weka.clusterers.forOPTICSAndDBScan.DataObjects.EuclideanDataObject")).
                    getConstructor(new Class[]{Instance.class, String.class, Database.class});
            o = co.newInstance(new Object[]{instance, key, database});
        } catch (NoSuchMethodException e) {
            e.printStackTrace();
        } catch (SecurityException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        } catch (InstantiationException e) {
            e.printStackTrace();
        } catch (IllegalAccessException e) {
            e.printStackTrace();
        } catch (InvocationTargetException e) {
            e.printStackTrace();
        }

        return (DataObject) o;
    }
	
    public Database databaseForName(Instances instances) {
        Object o = null;

        Constructor co = null;
        try {
            co = (Class.forName("weka.clusterers.forOPTICSAndDBScan.Databases.SequentialDatabase")).getConstructor(new Class[]{Instances.class});
            o = co.newInstance(new Object[]{instances});
        } catch (NoSuchMethodException e) {
            e.printStackTrace();
        } catch (SecurityException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        } catch (InstantiationException e) {
            e.printStackTrace();
        } catch (IllegalAccessException e) {
            e.printStackTrace();
        } catch (InvocationTargetException e) {
            e.printStackTrace();
        }

        return (Database) o;
    }
    
	@Override
	public void setNumClusters(int numClusters) throws Exception {
		this.numClusters=numClusters;
		
	}
	public void setMinPoints(int minPoints) throws Exception {
		this.minPoints=minPoints;
		
	}
	public void setEpsilon(double epsilon) throws Exception {
		this.epsilon=epsilon;
		
	}
	public void setMaxIterations(int maxIterations) throws Exception {
		this.maxIterations=maxIterations;
		
	}

	public int getNumClusters() throws Exception {
		return this.numClusters;
		
	}
	
	public int getMinPoints() throws Exception {
		return this.minPoints;
		
	}
	public double getEpsilon() throws Exception {
		return this.epsilon;
		
	}
	public double getMaxIterations() throws Exception {
		return this.maxIterations;
		
	}

	
    public Enumeration<Option> listOptions() {
        Vector<Option> vector = new Vector<Option>();

        vector.addElement(
                new Option("\tepsilon (default = 0.9)",
                        "E",
                        1,
                        "-E <double>"));
        vector.addElement(
                new Option("\tnumClusters (default = 2)",
                        "N",
                        1,
                        "-N <int>"));
        vector.addElement(
                new Option("\tminPoints (default = 6)",
                        "M",
                        1,
                        "-M <int>"));
        vector.addElement(
                new Option("\tmaxIterations (default = 300)",
                        "I",
                        1,
                        "-I <int>"));
        return vector.elements();
    }
    
    public void setOptions(String[] options) throws Exception {
        String optionString = Utils.getOption('E', options);
        if (optionString.length() != 0) {
            setEpsilon(Double.parseDouble(optionString));
        }
        
        optionString = Utils.getOption('N', options);
        if (optionString.length() != 0) {
            setNumClusters(Integer.parseInt(optionString));
        }
        
        optionString = Utils.getOption('M', options);
        if (optionString.length() != 0) {
            setMinPoints(Integer.parseInt(optionString));
        }
        
        optionString = Utils.getOption('I', options);
        if (optionString.length() != 0) {
            setMaxIterations(Integer.parseInt(optionString));
        }

    }
    
	@SuppressWarnings({ "unchecked", "rawtypes" })
	public String[] getOptions () {
		int       	i;
		Vector    	result;
		String[]  	options;

		result = new Vector();
		
		result.add("-E");
		try {
			result.add("" + this.getEpsilon());
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		result.add("-M");
		try {
			result.add("" + this.getMinPoints());
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		result.add("-N");
		try {
			result.add("" + this.getNumClusters());
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		result.add("-I");
		try {
			result.add("" + this.getMaxIterations());
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		options = super.getOptions();
		for (i = 0; i < options.length; i++)
			result.add(options[i]);

		return (String[]) result.toArray(new String[result.size()]);	  
	}


}
