#%%

import csv
import networkx as nx
import math
import time
from datetime import datetime
from geopy import distance

def distance_sphere(lat1,lng1,lat2,lng2):
    arc=distance.distance((lat1,lng1), (lat2,lng2)).km
    return arc #get meters

def dfs_allpaths_iter(G, source, cutoff=None):
    """
    Produce all paths from source to target in
    a depth-first-search starting at source.

    Pre-condition: source not equal to target

    Parameters:
        G: undirected graph
        source: source node
        cutoff: max number of edges of a given path.
                if this parameter is None, then the
                function produces all possible paths
                between source and target.
    """
    if cutoff==None:#8 min default searching time
        cutoff = 8
    visited = [source]
    stack = [iter(G[source])]
    while stack:
        children = stack[-1]
        child = next(children, None)
        if child == None:
            stack.pop()
            visited.pop()
        elif len(visited) >= cutoff:
            yield (visited)
            stack.pop()
            visited.pop()
        elif len(visited)<cutoff:
            visited.append(child)
            stack.append(iter(G[child]))
        yield (visited)
#------
def construct_GT(n0,G,this_index,this_endtime,cutoff):
    """
        Construct Graph Tree according to start searching node and time,
        alternatively, construct network properties as a file input
    """

    global indexx
    if this_index==0:
        GT.add_node(this_index)
        GT.nodes[this_index]['lat']=G.nodes[n0]['lat']
        GT.nodes[this_index]['lng']=G.nodes[n0]['lng']
        GT.nodes[this_index]['color']=G.nodes[n0]['color']
        GT.nodes[this_index]['endtime']=this_endtime
        GT.nodes[this_index]['child_list']=G.neighbors(n0)

    for row in G.neighbors(n0):
        if row!=n0:
            indexx+=1
            this_distance=distance_sphere(G.nodes[n0]['lat'],G.nodes[n0]['lng'],G.nodes[row]['lat'],G.nodes[row]['lng'])
            this_time=this_distance/3*60
            this_endtime1=this_endtime+this_time
            GT.add_node(indexx)
            GT.nodes[indexx]['lat']=G.nodes[n0]['lat']
            GT.nodes[indexx]['lng']=G.nodes[n0]['lng']
            GT.nodes[indexx]['color']=G.nodes[row]['color']
            GT.nodes[indexx]['endtime']=this_endtime1
            GT.nodes[indexx]['child_list']=G.neighbors(row)
            GT.add_edge(this_index,indexx)
            GT[this_index][indexx]['lat1']=G[n0][row]['lat1']
            GT[this_index][indexx]['lng1']=G[n0][row]['lng1']
            GT[this_index][indexx]['lat2']=G[n0][row]['lat2']
            GT[this_index][indexx]['lng2']=G[n0][row]['lng2']
            GT[this_index][indexx]['pick_p']=G[n0][row]['pick_p']
            if this_endtime>cutoff:
                GT[this_index][indexx]['seg_g']=G[n0][row]['seg_g']*(this_endtime-cutoff)/this_time
            else:
                GT[this_index][indexx]['seg_g']=G[n0][row]['seg_g']
                construct_GT(row,G,indexx,this_endtime1,cutoff)

#-----------------------------------------------------------------------------
def calculat_total_g(G,path):
    total_gg=0
    pickup_p=1
    nopick_p=0
    for n in range(0,len(path)-1):
        total_gg=total_gg+(G[path[n]][path[n+1]]['seg_g']*G[path[n]][path[n+1]]['pick_p']-0.18*distance_sphere(G.nodes[path[n]]['lat'],G.nodes[path[n]]['lng'],G.nodes[path[n+1]]['lat'],G.nodes[path[n+1]]['lng'])*(1-G[path[n]][path[n+1]]['pick_p']))*pickup_p
        pickup_p=(1-G[path[n]][path[n+1]]['pick_p'])*pickup_p
    return total_gg

def dfs_allpaths(G,source,cutoff=None):
    '''
    Produce all paths in a depth-first-search starting at source.
    '''
    i=0
    max_fair=0
    max_fair_index=0
    path_iter = dfs_allpaths_iter(G,source,cutoff)
    max_path=[source]*cutoff

    for path in path_iter:
        status=1
        if len(path)>2:
            for i in range(0,len(path)-2):
                if path[i]==path[i+2]:
                    status=0
                    break
        if status==1:
            total_g=calculat_total_g(G,path)
            if total_g>max_fair:
                max_fair=total_g
                max_path=[0]*cutoff
                for j in range(0,len(path)):
                    max_path[j]=path[j]
    return (max_fair,max_path)

def distance1(lat,lng,lat1,lng1,lat2,lng2): #calculate distance of driver's current loa to segment midpoint
    latt=(float(lat1)+float(lat2))/2
    lngg=(float(lng1)+float(lng2))/2
    return((float(lat)-latt)**2+(float(lng)-lngg)**2)

def start_searching_seg(G,la,ln):
    t=100000;
    for n1,n2 in G.edges():
        if distance1(la,ln,G[n1][n2]['lat1'],G[n1][n2]['lng1'],G[n1][n2]['lat2'],G[n1][n2]['lng2'])<t:
            t=distance1(la,ln,G[n1][n2]['lat1'],G[n1][n2]['lng1'],G[n1][n2]['lat2'],G[n1][n2]['lng2'])
            t1=n1
            t2=n2
    return(t2)


G=nx.Graph() #creat a Graph
i=0
with open ('../data/refined_vertice.csv', 'r') as f: #create graph nodes
    reader = csv.reader(f)
    for row in reader:
        G.add_node(int(row[0]))
        G.nodes[int(row[0])]['lat']=float(row[1])
        G.nodes[int(row[0])]['lng']=float(row[2])
        G.nodes[int(row[0])]['color']='white'
        G.nodes[int(row[0])]['total_g']=0

j=0
with open ('../data/refined_edges.csv', 'r') as f: #create graph edge (adjacent segs)
    reader = csv.reader(f)
    for row in reader:
        i=int(row[1])
        j=int(row[2])
        if i<j:
            G.add_edge(i,j,lat1=G.nodes[i]['lat'],lng1=G.nodes[i]['lng'],lat2=G.nodes[j]['lat'],lng2=G.nodes[j]['lng'],seg_g=0,pick_p=0,N_0=0,N_1=1,fair=0,wait_time=[])


with open('../data/network_properties.csv','r') as f1:
    reader1 = csv.reader(f1)
    for row in reader1:
        a=int(row[0])
        b=int(row[1])
        G.add_edge(a,b)
        G[a][b]['lat1']=float(row[2])
        G[a][b]['lng1']=float(row[3])
        G[a][b]['lat2']=float(row[4])
        G[a][b]['lng2']=float(row[5])
        G[a][b]['pick_p']=float(row[8])
        G[a][b]['seg_g']=float(row[9])

#searching range
min_lat=39.86
min_lng=116.31
max_lat=39.967
max_lng=116.46
j=0


#given a query from taxi 20384, located at coordinate "39.9005705,116.4966436", which got a pickup at '22:57:46':

query=['20384', '22:57:35', '39.9005705', '116.4966436', '39.9005743', '116.4955831', '39.8984675', '116.484878', '12/1/2013', '22:57:46', '12/1/2013', '22:59:30', '17.7900534\n']


start_lat=float(query[2])
start_lng=float(query[3])
hour=int(query[1].split(':')[0])
indexx=0


n=start_searching_seg(G,start_lat,start_lng)
CPUtime=time.process_time()
# GT=nx.Graph()
start_time = datetime.strptime(query[8] + "-" + query[1], "%m/%d/%Y-%H:%M:%S")
end_time = datetime.strptime(query[8] + "-" + query[9], "%m/%d/%Y-%H:%M:%S")
realtime = (end_time - start_time).seconds / 60

# construct_GT(n,G,0,0,realtime)
(fair,pathss)=dfs_allpaths(G,n,10)
CPUtime=time.process_time()-CPUtime
# while pathss[-1] == 0 and len(pathss) > 2:
#     del pathss[-1]
ddistance=0
drive_route=[[G.nodes[pathss[i]]['lat'],G.nodes[pathss[i]]['lng']] for i in range(len(pathss))]
for i in range(len(pathss)-1):
    ddistance =ddistance+distance_sphere(G.nodes[pathss[i]]['lat'],G.nodes[pathss[i]]['lng'],G.nodes[pathss[i+1]]['lat'],G.nodes[pathss[i+1]]['lng'])

print('query starts at:',start_lat,start_lng)
print('suggest route:',drive_route)
print('estimate earning:',fair)

writer=csv.writer(open('../results/query_response.csv','w'))
writer.writerow(['query starts at:',start_lat,start_lng])
writer.writerow(['suggest route:']+drive_route)
writer.writerow(['estimate earning:',fair])
