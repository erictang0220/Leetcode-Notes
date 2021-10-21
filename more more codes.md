## Breadth First Search: Shortest Reach (Hackerrank)
* Consider an undirected graph where each edge weighs 6 units. Each of the nodes is labeled consecutively from 1 to n. You will be given a number of queries. For each query, you will be given a list of edges describing an undirected graph. After you create a representation of the graph, you must determine and report the shortest distance to each of the other nodes from a given starting position using the breadth-first search algorithm (BFS). Return an array of distances from the start node in node number order. If a node is unreachable, return -1 for that node.
* Subtle logic flaw in my intitial implementation --> should mark the node visited in the for loop!!!
* bfs --> use queue
```c++
vector<int> bfs(int n, int m, vector<vector<int>> edges, int s) {
	vector<int> ans(n-1, -1);
	vector<vector<int>> adjList(n+1); // 1-based index
	vector<bool> visited(n+1, false);
	
	// create the adjacency list
	for(int i=0; i < edges.size(); i++) {
		adjList[edges[i][0]].push_back(edges[i][1]);
		adjList[edges[i][1]].push_back(edges[i][0]);
	}

	queue<int> q;
	q.push(s);
	visited[s] = true;
	
	while(!q.empty()) {
		int tmp = q.front();
		q.pop();
		for(int k : adjList[tmp]) {
			if(!visited[k]) {
				if(tmp == s) {
					if(k > s) ans[k-2] = 6;
					else ans[k-1] = 6;
				}
				else if(tmp > s){
					if(k > s) ans[k-2] = ans[tmp-2] + 6;
					else ans[k-1] = ans[tmp-2] + 6;
				}
				else {
					if(k > s) ans[k-2] = ans[tmp-1] + 6;
					else ans[k-1] = ans[tmp-1] + 6;
				}
				// should mark visited here!!! if marked before the for loop, some ans values can be overwritten!!! (e.g. 1,2,3 triagular graph)
				visited[k] = true;
				q.push(k);
			}
		}
	}
	return ans;
}
```
## Insert Interval
* You are given an array of non-overlapping intervals intervals where intervals[i] = [starti, endi] represent the start and the end of the ith interval and intervals is sorted in ascending order by starti. You are also given an interval newInterval = [start, end] that represents the start and end of another interval.
* Insert newInterval into intervals such that intervals is still sorted in ascending order by starti and intervals still does not have any overlapping intervals (merge overlapping intervals if necessary).
* Trickly logic!!
```c++
vector<vector<int>> insert(vector<vector<int>>& intervals, vector<int>& newInterval) {
	vector<vector<int>> ret;
	int i=0, n=intervals.size();
	
	// if the entire newInterval is above intervals[i], no need to change
	while(i < n && newInterval[0] > intervals[i][1]) {
		ret.push_back(intervals[i]);
		i++;
	}

	int start=newInterval[0], end = newInterval[1];
	
	// CRUCIAL!!!
	// upper bound of newInterval >= lower bound of interval —> overlap, since already eliminate the case where newInterval is entirely above interval, implying newInterval[0] < intervals[i][1]
	while(i < n && end >= intervals[i][0]) {
		start = min(intervals[i][0], start);
		end = max(intervals[i][1], end);
		i++;
	}
	
	ret.push_back({start, end});
	
	// push back intervals[i] that are entirely above newInterval
	while(i < n) {
		ret.push_back(intervals[i]);
		i++;
	}
	return ret;
}

// using lambda in sorting
vector<vector<int>> insert(vector<vector<int>>& intervals, vector<int>& newInterval) {
	intervals.push_back(newInterval);
	
	// sort based on the lower bounds of each interval
	sort(intervals.begin() , intervals.end() , [](vector<int> &i1 , vector<int> &i2){return i1[0] < i2[0];});

	vector<vector<int>> ans;
	for(vector<int> v : intervals) {
		if(ans.empty()) ans.push_back(v);
		// must overlap if satisfied the if condition, since lower bounds already sorted in ascending order
		else if(ans.back()[1] >= v[0]) {
			ans.back()[1] = max(ans.back()[1], v[1]);
		}
		// else the entire interval v is above the previous interval
		else ans.push_back(v);
	}
	return ans;
}
```
## Dijkstra: Shortest Reach 2 (Hackerrank)
* Dijkstra used to find the shortest path from a node to other nodes
* Good video for Dijkstra: https://www.youtube.com/watch?v=_lHSawdgXpI
* Given an undirected graph and a starting node, determine the lengths of the shortest paths from the starting node **s** to all other nodes in the graph. If a node is unreachable, its distance is -1.
* ALWAYS use priority queue for dijkstra --> requried to find unvisited vertex with minimum distance or else get at least one of the edges wrong.
	* Use queue instead of priority queue in the following example, path to 4 is 13 instead of 8
	4 4

	1 2 8

	2 4 5

	1 3 2

	3 2 1

	1—2—4

	\ /

	3
* Default C++ logic of pair comparison: a > b is true if a.first > b.first or (a.first==b.first && a.second > b.second)
```c++
vector<int> shortestReach(int n, vector<vector<int>> edges, int s) {
	vector<int> ret(n+1, INT_MAX);
	vector<vector<pair<int,int>>> adjList(n+1); 
	vector<bool> mark(n+1, 0);

	// put weight first so priority queue can sort it in ascending order
	for(int i=0; i < edges.size(); i++) {
		adjList[edges[i][0]].push_back({edges[i][2], edges[i][1]});
		adjList[edges[i][1]].push_back({edges[i][2], edges[i][0]});
	}
	
	// pairs with smaller weight are in front of the queue (sorted in increasing order)
	priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> q;
	ret[s]=0;
	// it takes 0 distance from node s to s
	q.push({0,s});

	while(!q.empty()) {
		pair<int, int> x = q.top();
		// w is the distance from s to 'start' node
		int start = x.second, w = x.first;
		q.pop();
		// use mark so push the same node repeatedly, doesn't mean that its value can't get changed later (it can appear as k.second)
		if (mark[start]) continue;
		mark[start] = true;
		for(pair<int,int> k : adjList[start]) {
			if(ret[k.second] > w + k.first) {
				ret[k.second] = w + k.first;
				if (!mark[k.second]) q.push({ret[k.second], k.second});
			}
		}
	}

	vector<int> ans;
	for(int i=1; i <= n; i++) {
		if(i==s) continue;
		if(ret[i]==INT_MAX) ans.push_back(-1);
		else ans.push_back(ret[i]);
	}

	return ans;
}
```
## Prim's (Minimal spanning tree (MST)) : Special Subtree (Hackerrank)
* prim's algorithm used to find the minimal spanning tree (no cycle)
* One specific node is fixed as the starting point of finding the subgraph using Prim's Algorithm.
Find the total minimal weight or the sum of all edges in the subgraph. (Techinically the starting point doesn't matter)
* Good video for Prim's algorithm: https://www.youtube.com/watch?v=cplfcGZmX7I
```c++
int prims(int n, vector<vector<int>> edges, int s) {
	vector<vector<pair<int,int>>> adjList(n+1); 
	vector<bool> mark(n+1, 0);

	// put weight first so priority queue can sort it in ascending order
	for(int i=0; i < edges.size(); i++) {
		adjList[edges[i][0]].push_back({edges[i][2], edges[i][1]});
		adjList[edges[i][1]].push_back({edges[i][2], edges[i][0]});
	}
	
	
	// ensures edges with minimal weight is put in front
	priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> q;
	// start at the node with zero weight
	q.push({0,s});

	long ans=0;
	while(!q.empty()) {
		pair<int, int> x = q.top();
		// w is the distance from s to 'start' node
		int start = x.second, w = x.first;
		q.pop();
		if (mark[start]) continue;
		mark[start] = true;
		ans+=w;

		for(pair<int,int> k : adjList[start]) {
			if(!mark[k.second]) q.push(k);
		}
	}
	return ans;
}
```
## Kruskal (MST): Really Special Subtree
* Serve the same purpose as Prim's algorithm, but different implementation
	* Prim's algorithm builds up the tree greedily, while Kruskal's algorithm greedily selects edges, maintaining a forest.
* ![dif btw Prims and Kruskal](https://user-images.githubusercontent.com/50003319/131259565-55c15a1d-775d-4954-9feb-76a2f230a7f3.jpg)
* Good video for Kruskal: https://www.youtube.com/watch?v=71UQH7Pr9kU
* Kruskal requires priority queue (like Prim's) and **union find** (unlike Prim's)!!!
* If simply use visited approach, the following case wouldn't work
	4 5
	1 2 1
	3 2 150
	4 3 99
	1 4 100
	3 1 200
```c++
vector<int> id;

// every node is separated, represented by its index
void initialize(int n){
	for(int i=0; i < n; i++)
		id.push_back(i);
}

// return the root of node x
int root(int x){
	// while pointing to some other node (not at root)
	while(id[x]!=x){
		// make the children point upward by one level!!
		id[x] = id[id[x]];
		x = idx[x];
	}
	return x;
}

// join the two roots
void union1(int x,int y){
	int p = root(x);
	int q = root(y);
	// make one root point to the other
	id[p] = id[q];
}

int kruskals(int g_nodes, vector<int> g_from, vector<int> g_to, vector<int> g_weight) {
	initialize(g_nodes+1);
	vector<vector<tuple<int,int,int>>> adjList(g_nodes+1);
	vector<bool> vis(g_nodes+1, false);
	priority_queue<tuple<int,int,int>, vector<tuple<int,int,int>>, greater<tuple<int,int,int>>> q;
	for(int i=0; i < g_from.size(); i++) {
		q.push({g_weight[i],g_from[i],g_to[i]});
	}

	long minSum=0;

	while(!q.empty()) {
		int w=get<0>(q.top()), start=get<1>(q.top()), end=get<2>(q.top());
		q.pop();
		// if already connected, skip
		if(root(start)==root(end)) continue;
		// else add the new edge
		minSum+=w;
		// connect the two nodes
		union1(start,end);
	}
	return minSum;
}
```
## Copy List with Random Pointer
* A linked list of length n is given such that each node contains an additional random pointer, which could point to any node in the list, or null. Make a deep copy of the list
* second approach intuition (use 1->2->3 example for better understanding):
	1. create a deep copy of the linked list on the next node of the current node (zigzag shape if two lists placed parallel)
	2. assign corresponding deep-copied random nodes to deep-copied nodes
	3. separate the original list from the deep copy list (share the same nullptr)  
```c++
// Using a hashmap, time complexity O(N), space complexity O(N)
Node* copyRandomList(Node* head) {
	unordered_map<Node*,Node*> m;
	Node* iter = head;
	while(iter){
		Node* temp = new Node(iter->val);
		m[iter] = temp;
		iter = iter->next;
	}

	//STEP 2 : UPDATE THE NEXT AND RANDOM POINTERS ACCORDINGLY 
	iter = head;
	while(iter){
		// didn't intialize m[nullptr], access it will return the "null value" of a pointer, which is nullptr!!!
		m[iter]->next = m[iter->next];
		m[iter]->random = m[iter->random];
		iter = iter->next;
	}

	return m[head];
}

// Using crazy logic! time complexity O(N), space complexity O(1)
Node* copyRandomList(Node* head) {
	Node* iter=head, *front;
	while(iter) {
		front = iter->next;
		Node* tmp = new Node(iter->val);
		iter->next = tmp;
		tmp->next = front;
		iter = front;
	}

	iter=head;
	while(iter) {
		if(iter->random) {
			// set the corresponding random nodes of copied nodes
			iter->next->random = iter->random->next;
		}
		// skip the copied nodes
		iter=iter->next->next;
	}

	iter=head;
	Node* dummy = new Node(0);
	Node* deepCopy = dummy;
	while(iter) {
		front = iter->next->next;
		deepCopy->next = iter->next; // separation
		deepCopy = deepCopy->next;
		iter->next = front; //separation
		iter = front;
	}
	return dummy->next;
}
```
## Count Complete Tree Nodes
* Given the root of a **complete** binary tree, return the number of the nodes in the tree.
* Try to do it < O(N)
```c++
// O(N) 
int countNodes(TreeNode *root) {
	if(!root) return 0;
	return 1 + countNodes(root->left) + countNodes(root->right); 
}

// height O(logN), countNodes O(logN)

int height(TreeNode *root) {
	if(!root) 
		return -1;
	return 1 + height(root->left);
}

int countNodes(TreeNode *root) {
	int h = height(root); // number of branches on the far left
	// root is a nullptr, h == -1
	if(h < 0) 
		return 0;
	// last node on the right subtree!!! h - 1 (root node branch)
	// # of nodes on the left subtree = (1 << h) - 1, plus root = (1 << h) 
	else if(height(root->right) == h-1)
		return (1 << h) + countNodes(root->right);
	// last node on the left subtree
	// # of nodes on the right subtree = (1 << (h-1)) - 1, plus root = (1 << (h-1)) 
	else 
		return (1<< (h-1)) + countNodes(root->left);
}
```
## Floyd : City of Blinding Lights (Hackerrank)
* Given a directed weighted graph where weight indicates distance, for each query, determine the length of the shortest path between nodes. There may be many queries, so efficiency counts.
* Use an adjacency matrix
* Use Floyd–Warshall algorithm: **shortest path between all pairs of vertices, negative edges allowed**
	* IMPORTANT: the order of the three for loops matters --> use the intermediate node as the outermost loop variable **k**, represents the shortest possible path from i to j using vertices only from the set of nodees {1,2,...,k} as intermediate points along the way.
* Good video for Floyd-Warshall: https://www.youtube.com/watch?v=4OQeCuLYj-4
```c++
vector<int> road_from(road_edges);
vector<int> road_to(road_edges);
vector<int> road_weight(road_edges);
// can't pick infinity will overflow, but pick a large enough number ( > #edges * max_possible_weight)
vector<vector<int>> ans(road_nodes+1, vector<int>(road_nodes+1, 100000));

// distance to itself is 0
for(int i=1; i <= road_nodes; i++) {
	ans[i][i] = 0;
}

// add the graph
for(int i=0; i < road_edges; i++) {
	ans[road_from[i]][road_to[i]] = road_weight[i];
}

for(int i=1; i <= road_nodes; i++) {
	for(int j=1; j <= road_nodes; j++) {
		// if(ans[j][i] == 1000) continue;
		for(int k=1; k <= road_nodes; k++) {
			if(ans[j][k] > ans[j][i] + ans[i][k]) {
				ans[j][k] = ans[j][i] + ans[i][k];
			}
		}
	}
}

// x and y are queries (from x to y)
cout << (ans[x][y]==100000? -1 : ans[x][y]) << endl;
```
## Even Tree (Hackerrank)
* Find the maximum number of edges you can remove from the tree to get a forest such that each connected component of the forest contains an even number of nodes.
* root is at **1**!!! so t_from[i] always less than t_to[i]
```c++
struct Node{
	int parent,children=0;
};

int evenForest(int t_nodes, int t_edges, vector<int> t_from, vector<int> t_to) {
	int ans=0;
	vector<Node> tree(t_nodes+1);
	// tree[i].children accounts for the # of IMMEDIATE chilren of the node
	for(int i=0; i < t_edges; i++) {
		tree[t_from[i]].parent = t_to[i];
		tree[t_to[i]].children++;
	}
	
	// not counting the root bc it has no edge with a previous node to sever
	for(int i=t_nodes; i > 1; i--) {
		// odd number of children —> even number of nodes including itself
		if(tree[i].children % 2 == 1) {
			ans++;
			// edge severed, so decrement the # of children
			tree[tree[i].parent].children--;
		}
	}
	return ans;
}

// another approach: track the cumulative children for each node (including itself)
struct Node{
	int num,root;
};

int evenForest(int t_nodes, int t_edges, vector<int> t_from, vector<int> t_to) {
	int count = 0;

	vector<Node> nodes = vector<Node>(t_nodes);    
	for(int i = 0; i < t_nodes; i++){
		// just the node itself so num == 1 
		nodes[i].num = 1;
		nodes[i].root = -1;
	}
	
	for(int i = 0; i < t_edges; i++){
		nodes[t_from[i]-1].root = t_to[i]-1;   
	}

	for(int i = t_nodes-1; i > 0; i--) {
		// sum up the previous children
		if(nodes[i].root >= 0)
			nodes[nodes[i].root].num += nodes[i].num;
	}

	for(int i = 0; i < t_nodes; i++) {
		// not root and has even # of children (including itself)
		if(nodes[i].root >= 0 && nodes[i].num % 2 == 0) count++;
	}
	
	return count;
}
```
## Snakes and Ladders: The Quickest Way Up (Hackerrank)
* Return an integer that represents the minimum number of moves required.
* Rules:
	1. Starting from square 1, land on square 100 with the exact roll of the die. If moving the number rolled would place the player beyond square 100, no move is made.
	2. If a player lands at the base of a ladder, the player must climb the ladder. Ladders go up only.
	3. If a player lands at the mouth of a snake, the player must go down the snake and come out through the tail. Snakes go down only.
* NOTE: indices on the board is labelled in zigzag manner
* IDEA: use graph and bfs(??but without queue)
```c++
void breadthFirstSearch(vector<vector<int>>& adjList, int vertices, int level[]) {
	vector<int> temp;
	int i, lev, flag = 1;
	// 'lev' represents the level to be assigned
	// 'flag' used to indicate if graph is exhausted, if so then it will never be set to 1 in the for loop

	lev = 0;
	level[1] = lev;
	// We start from index 1 so it's at level 0. All immediate neighbours (2,....,7) are at level 1, 8,..., 13 are at level 2 and so on.
	// note only 2~7 are have parent index 1, others have parent of index (i - 6) --> don't need parent array for this problem

	while (flag) {
		flag = 0; 
		for (i = 1; i <= vertices; ++i) {
			// search from 1...100 to check the level
			if (level[i] == lev) {
				flag = 1;
				temp = adjList[i];

				for(int j=0; j < temp.size(); j++) {
					// if set then don't overwrite, the ones set first are the least level possible (desired)
					if (level[temp[j]] != -1) {
						continue;
					}
					
					level[temp[j]] = lev + 1;
				}
			}
		}
		// finish all nodes with the current lev (current breadth?), increment to go to the next level
		++lev;
	}
}

// replace num in V with replaceNum
void replace(vector<int>& v, int num, int replacedNum) {
	auto it = find(v.begin(), v.end(), num);
	if(it != v.end()) {
		v[it-v.begin()] = replacedNum;
	}
}

int main() {
	int t;	// Test cases

	scanf("%d", &t);

	while (t--) {
		int vertices, edges, i, j, v1, v2;

		vertices = 100;		// Assume it is a 100 checks board
		edges = 0;

		vector<vector<int>> adjList(vertices+1);

		int level[vertices + 1];
		// Each element of Level Array holds the Level value of that node

		// Initialising our arrays
		for (i = 0; i <= vertices; ++i) {
			level[i] = -1;
		}

		// Add normal edges representing movements by dice
		for (i = 1; i <= vertices; ++i) {
			for (j = 1; j <= 6 && j + i <= 100; ++j) {
				adjList[i].push_back(i+j);
			}
			// reverse it so the bigger numbers are in front
			reverse(adjList[i].begin(), adjList[i].end());
		}

		int numLadders, numSnakes;
		char temp;

		scanf("%d", &numLadders);

		// Ladder Edges
		for (i = 0; i < numLadders; ++i) {
			scanf("%d%c%d", &ladders[i][0], &temp, &ladders[i][1]);
			
			// only max of 6 indices in front of the index of the bottom of the ladder contain ladders[i][0]
			j = ladders[i][0] - 6;

			if (j < 1) {
				j = 1;
			}

			for (; j < ladders[i][0]; ++j) {
				replace(adjList[j], ladders[i][0], ladders[i][1]);
			}
		}

		scanf("%d", &numSnakes);

		// Snakes Edges
		for (i = 0; i < numSnakes; ++i) {
			scanf("%d%c%d", &snakes[i][0], &temp, &snakes[i][1]);

			// only max of 6 indices in front of the index of the head of the snake contain snakes[i][0]
			j = snakes[i][0] - 6;

			if (j < 1) {
				j = 1;
			}

			for (; j < snakes[i][0]; ++j) {
				replace(adjList[j], snakes[i][0], snakes[i][1]);
			}
		}

		breadthFirstSearch(adjList, vertices, level);
		printf("%d\n", level[vertices]);
	}

	return 0;
}
```
## Journey to the Moon (Hackerrank)
* The member states of the UN are planning to send 2 people to the moon. They want them to be from different countries. You will be given a list of pairs of astronaut ID's. Each pair is made of astronauts from the same country. Determine how many pairs of astronauts from different countries they can choose from.
* NEAT TRICK: 
	1. Set A has a elements.
	Answer = 0 (Since I don't have another country to pair with)
	2. Set A has a elements. Set B with b elements.
	Answer = a x b;
	3. A, B, C --> a, b, c elements
	Answer = (a x b) + (a x c) + (b x c) [because we can select a pair from A and B, or A and C or B and C]
	can be written as answer = (a x b) + c x (a + b)
	4. A, B, C, D --> a, b, c, d elements
	Answer = (a x b) + (a x c) + (a x d) + (b x c) + (b x d) + (c x d) = (a x b) x (a + b) x c + (a + b + c) x d
	5. new set **S** with s elements --> number of ways to (previous sum) x  s + previous answer
```c++
vector<bool> vis;
void dfs(vector<vector<int>>& astronaut, long long& same, int start) {
	vis[start] = true;
	same++;
	for(int i=0; i < astronaut.size(); i++) {
		// two if statements to account for start node on either end of edges
		if(astronaut[i][0] == start && !vis[astronaut[i][1]]) {
			dfs(astronaut, same, astronaut[i][1]);
		}
		if(astronaut[i][1] == start && !vis[astronaut[i][0]]) {
			dfs(astronaut, same, astronaut[i][0]);
		}
	}
}

long long journeyToMoon(int n, vector<vector<int>> astronaut) {
	vis.resize(n, false);
	long long ans=0, sum=0;
	for(int i=0; i < n; i++) {
		long long tmp = 0;
		if(!vis[i]) {
			dfs(astronaut, tmp, i);
			// apply the logic above, note when there's only one contry, ans == 0
			ans += tmp*sum;
			sum += tmp;
		}
	}
	return ans;
}
```
## Maximal Square
* Given an m x n binary matrix filled with 0's and 1's, find the largest square containing only 1's and return its area.
* DP!! Find the largest side length of the square we can form at dp[i][j]
* min({x,y,z}) finds the minimum of the three!
* If all 1's in 3 x 3, dp = 
1	1	1
1	2	2
1	2	3
```c++
int maximalSquare(vector<vector<char>>& matrix) {
	int m=matrix.size(), n=matrix[0].size();
	int ans=0;
	vector<vector<int>> dp(m, vector<int>(n,0));
	for(int i=0; i < m; i++) {
		for(int j=0; j < n; j++) {
			if(matrix[i][j] == '1') {
				// current iteration at bottom right, only 1 x 1 square possible
				if(i == 0 || j == 0) 
					dp[i][j] = 1;
				else 
					// check the min of all three blocks!!! (top left --> bottom right(current iteration))
					dp[i][j] = 1+min({dp[i-1][j-1], dp[i][j-1], dp[i-1][j]});
				ans = max(ans, dp[i][j]);
			}
		}
	}
	return pow(ans,2);
}
```
## The Coin Change Problem (Hackerrank)
* Given an amount and the denominations of coins available, determine how many ways change can be made for amount. There is a limitless supply of each coin type.
* DP!!! 
```c++
long getWays(int n, vector<long> c) {
	// index 0 is base case for any existing coins
	vector<long> dp(n+1,0);
	dp[0] = 1;
	// each iteration adds to the number of ways of using that coin only
	for(int i=0; i < c.size(); i++) {
		// from the current coin value to target (or else index becomes negative), sum the number of ways
		for(int j=c[i]; j < n+1; j++) {
			// one coin away from the current j
			// first iteration always add dp[0], which is 1, the coin can represent its value
			dp[j] += dp[j-c[i]];
		}
	}
	return dp[n];
}
```
## Cisco interview
* Find the number of ways to interpret a string of digits in letters ('0' is a, '1' is b, ..... '25' is z)
```c++
int traverse(string start, string rem) {
	if(start.size() == 2) {
		if(start[0] == '0' || stoi(start) >= 26) 
		return 0;
	}
	if(rem == "" || rem.size() == 1) return 1;
	return traverse(rem.substr(0,2), rem.substr(2)) + traverse(rem.substr(0,1), rem.substr(1));
}

string validInterpretations (string decInput) {
	string  answer;
	int tmp = traverse("", decInput);
	answer = to_string(tmp);
	return answer;
}
```
## Equal (Hackerrank) tricky!!!!
* To make things difficult, she must equalize the number of chocolates in a series of operations. For each operation, she can give **1, 2, 5** pieces to all but one colleague. Everyone who gets a piece in a round receives the same number of pieces.
* Intuition: instead of thinking adding chocolate to all but one person, thinking about ***taking away 1, 2, or 5 chocolates from one person***
* FURTHER intuition: 
	* sometimes it might take fewer total operations to convert all numbers to the (minimum in the array, m, -1) or m-2, m-3, m-4
	* f(m-5) takes N (size of array) more operations than f(m), therefore can't be a candidate
* TRICK: x = x / 5 + (x % 5) / 2 + (x % 5 % 2) --> minimum number of operations to reduce x to 0 by using 1, 2, 5!
```c++
int equal(vector<int> arr) {
	// m is the minimum in arr
	int m = *min_element(arr.begin(), arr.end());
	vector<int> t(4,0);
	for(int i=0; i < arr.size(); i++) {
		for(int j=0; j <= 4; j++) { 
			// consider cases where all elements are reduced to m-j
			// x represents the difference 
			int x = arr[i]-(m-j);
			// sum the min number of ops to reduce to m using 1,2,5
			x = x / 5 + (x % 5) / 2 + (x % 5 % 2);
			// sum the number of operations of all elements in arr in t[j]
			t[j]+=x;
		}
	}
	return *min_element(t.begin(), t.end());
}
```
## Maximum Subarray
* Tricky logic! Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.
* if sum ever reaches below 0, the sum starting from the following term will be greater whether it's negative or positive, set to 0 (in the process already capture the max before the current index)
```c++
int maxSubArray(vector<int>& nums) {
	int sum=0, ans=nums[0];
	for(int i=0; i < nums.size(); i++) {
		sum+=nums[i];
		ans = max(sum, ans);
		if(sum < 0) sum = 0;
	}
	return ans;
}
```
## Maximum Product Subarray
* Given an integer array nums, find a contiguous non-empty subarray within the array that has the largest product, and return the product.
* IDEA: maxVal and minVal keeps track of the max/min possible product **including** the element at index i (the previous terms can be included or not), product catches the actual max along the way
* Insight!! Still ensure contiguous product with max(nums[i], max(maxVal\*nums[i], minVal\*nums[i])) and min(nums[i], min(tmp\*nums[i], minVal\*nums[i])) since maxVal and minVal are contiguous product
```c++
int maxProduct(vector<int>& nums) {
	int minVal=nums[0], maxVal=nums[0], product=nums[0];
	for(int i=1; i < nums.size(); i++) {
		int tmp = maxVal;
		maxVal = max(nums[i], max(maxVal*nums[i], minVal*nums[i]));
		minVal = min(nums[i], min(tmp*nums[i], minVal*nums[i]));
		product = max(maxVal, product);
	}
	return product;
}
```
## Sherlock and Cost (Hackerrank)
* HARD!
* In this challenge, you will be given an array **B** and must determine an array **A**. There is a special rule: For all i, A[i] < B[i]. That is, A[i] can be any number you choose such that 1 <= A[i] <= B[i]. Your task is to select a series of A[i] given B[i] such that the sum of the absolute difference of consecutive pairs of A is maximized. 
* Intuition:
	* **Optimal solution must consist of either 1 or vi for each position!**
	* cur_lo = max cost using first i items of array v, ending with 1 at i th position. (1 denotes lo, note previous iterations can be vi or 1)
	* cur_hi = max cost for first i items of array v, ending in vi at i th position. 
	* current cur_lo/cur_hi each has two ways to reach: from previous cur_lo or cur_hi
	* For cur_lo
		* lo+lo_to_lo (which is just lo)
		* hi+hi_to_lo
	* For cur_hi
		* hi+hi_to_hi
		* lo+lo_to_hi
	* Take the max of last iteration of cur_lo and cur_hi 
```c++
int cost(vector<int> v) {
	int hi=0, lo=0;
	int hi_to_hi, lo_to_hi, hi_to_lo, cur_lo, cur_hi;
	for(int i=1; i < v.size(); i++) {
		hi_to_hi = abs(v[i-1] - v[i]);
		lo_to_hi = abs(1 - v[i]);
		hi_to_lo = abs(v[i-1] - 1);
		cur_lo = max(lo, hi+hi_to_lo);
		cur_hi = max(hi+hi_to_hi, lo+lo_to_hi);
		// lo/hi save the values for next iteration
		hi = cur_hi;
		lo = cur_lo;
	}
	return max(hi, lo);
}
```
## Metro Land Festival (Expedia)
* Metro Land is a country located on a 2D Plane. They are having a summer festival for everyone in the country and would like to minimise the overall cost of travel for their citizens. Costs of travel are calculated as abs(xi - x0) + abs(yi - y0). Determine the total cost of travel for all citizens to go to the festival at that location.
* Create weighted x and y (account for the number of people) then sort to figure the festival location (midway)!!!
```c++
int cost(x, y, a, b) {
	return (abs(x-a)+abs(y-b));
}

int greedy(vector<int> numpeople, vector<int> x, vector<int> y){
	vector<int> xx, yy;
	int ans = 0;
	for(int i = 0 ; i < numpeople.size();i++){
		int count = numpeople[i];
		while(count--) {
			xx.push_back(x[i]);
			yy.push_back(y[i]);
		}
	}

	sort(xx.begin(), xx.end());
	sort(yy.begin(), yy.end());
	int mx, my;

	mx = xx[xx.size() / 2];
	my = yy[yy.size() / 2];

	for(int i = 0; i < numpeople.size();  i++){
		ans += numpeople[i] * cost(mx, my, x[i], y[i]);
	}
	return ans;
}
```
## Count the number of ways to divide N in k groups incrementally (Expedia)
* DP!!!
* Given two integers N and K, the task is to count the number of ways to divide N into K groups of positive integers such that their sum is N and the number of elements in groups follows a non-decreasing order 
```c++
int dp[500][500][500];
long calc(int pos, int prev, int left, int k) {
	// pos represents number of elements
	// prev is the previous and current largest member
	// left is the number left to be added
	if(pos == k) {
		if(left == 0) return 1;
		else return 0;
	}
	
	// if divided into less/more than k groups
	if(left == 0) return 0;
	if(dp[pos][prev][left] != -1) return dp[pos][prev][left];


	int ans=0;
	for(int i=prev; i <= left; i++) {
		ans+=calc(pos+1, i, left-i, k);
	}

	return dp[pos][prev][left]=ans;
}

long countOptions(int people, int groups) {
	// cool way to set all elements to a number
	memset(dp, -1, sizeof(dp));
	return calc(0, 1, people, groups);
}
```
## Candies (Hackerrank)
* Alice is a kindergarten teacher. She wants to give some candies to the children in her class.  All the children sit in a line and each of them has a rating score according to his or her performance in the class.  Alice wants to give at least 1 candy to each child. If two children sit next to each other, then the one with the higher rating must get more candies. Alice wants to minimize the total number of candies she must buy.
* Two passes!! O(n), similar to the raining tank question
```c++
long candies(int n, vector<int> arr) {
	vector<long> v(n);
	v[0] = 1;
	for(int i=1; i < n; i++) {
		if(arr[i] > arr[i-1]) v[i] = v[i-1]+1;
		else v[i] = 1;
	}
	for(int i=n-2; i >= 0; i--) {
		// note the second condition of the if statement
		if(arr[i] > arr[i+1] && v[i+1] >= v[i]) v[i] = v[i+1]+1;
	}
	long sum=0;
	for(int i=0; i < n; i++) {
		sum+=v[i];
	}
	return sum;
}
```
## Reconstruct Original Digits from English 
* Given a string s containing an out-of-order English representation of digits 0-9, return the digits in ascending order.
* Idea: find the unique character for each digit!! If impossible subtract the number of digts that share the same character!
```c++
int main() {
	string s;
	cin >> s;
	vector<int> m(10, 0);
	for(char c : s) {
		if(c == 'z') m[0]++;
		if(c == 'o') m[1]++;
		if(c == 'w') m[2]++;
		if(c == 'h') m[3]++;
		if(c == 'u') m[4]++;
		if(c == 'f') m[5]++;
		if(c == 'x') m[6]++;
		if(c == 's') m[7]++;
		if(c == 'g') m[8]++;
		if(c == 'i') m[9]++;
	}

	m[1]-=(m[2]+m[4]+m[0]);
	m[3]-=m[8];
	m[5]-=m[4];
	m[7]-=m[6];
	m[9]-=(m[5]+m[6]+m[8]);

	string ans;
	for(int i=0; i < 10; i++) {
		while(m[i] > 0) {
			ans+=to_string(i);
			m[i]--;
		}
	}
	cout << ans << endl;
}
```
## Sam and substrings (Hackerrank)
* Given an integer as a string, sum all of its substrings cast as integers. As the number may become large, return the value modulo 10^9+7.
* Write out possible substrings to observe the patter!!!
* Example: 456, 4\*111\*1 + 5\*11\*2 + 6\*1\*3
```c++
long long substrings(string n) {
	long long int res = 0;
	long long int f = 1;
	int l=n.size();
	long long MOD = pow(10,9)+7;
	for(int i = l-1; i >= 0; i--) {
		res = (res + (n[i]-'0')*f*(i+1)) % MOD;
		f = (f*10+1) % MOD;
	}
	return res;
}
```
## House Robber II
* You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed. All houses at this place are arranged in a circle. That means the first house is the neighbor of the last one. Meanwhile, adjacent houses have a security system connected, and it will automatically contact the police if two adjacent houses were broken into on the same night.
* Trick!!!! Circular loop, can't consider both first and last elements --> do two separate calculations for each
```c++
int calc(vector<int> nums) {
	// first row is the maximum considering current element
	// second row is the maximum not considering current element
	vector<vector<int>> dp(2, vector<int>(nums.size(), 0));
	dp[0][0] = nums[0];
	dp[0][1] = nums[1];
	dp[1][1] = dp[0][0];
	for(int i=2; i < nums.size(); i++) {
		dp[0][i] = dp[1][i-1] + nums[i];
		dp[1][i] = max(dp[1][i-1], dp[0][i-1]);
	}
	return max(dp[0][nums.size()-1], dp[1][nums.size()-1]);
}

int rob(vector<int>& nums) {
	int n = nums.size();
	int maxi=0;
	if(n <= 3) {
		for(int i=0; i < n; i++) {
			maxi = max(maxi, nums[i]);
		}
		return maxi;
	}

	vector<int> tmp1(nums.begin(), nums.end()-1);
	vector<int> tmp2(nums.begin()+1, nums.end());
	return max(calc(tmp1), calc(tmp2));
}
```
## House Robber III
* The thief has found himself a new place for his thievery again. There is only one entrance to this area, called root. Besides the root, each house has one and only one parent house. After a tour, the smart thief realized that all houses in this place form a binary tree. It will automatically contact the police if two directly-linked houses were broken into on the same night. Given the root of the binary tree, return the maximum amount of money the thief can rob without alerting the police.
* Intuition: has to go bottom up from the tree (dfs)
* Cool dp with recursion and pair!!!
	* If current node not robbed, return the maximum value of the children nodes regardless they're robbed or not
	* If current node robbed, the children nodes can't be robbed and we add the current node value
```c++
pair<int,int> tra(TreeNode* root) {
	if(!root) return {0,0};
	pair<int,int> l = tra(root->left);
	pair<int,int> r = tra(root->right);
	int cur_not_rob = max(l.first,l.second) + max(r.first,r.second);
	int cur_rob = l.second + r.second + root->val;
	return {cur_rob, cur_not_rob};
}
int rob(TreeNode* root) {
	pair<int,int> ans = tra(root);
	return max(ans.first,ans.second);
}
```
## Linked List Cycle II
* Given the head of a linked list, return the node where the cycle begins. If there is no cycle, return null
* IDEA: if check cycle, do two pointers (slow, fast)!!!
* ![two pointers explain](https://user-images.githubusercontent.com/50003319/133544999-a155fc35-ebbe-44a2-a1ef-16b9647c4640.jpeg)
```c++
// two pointers!
ListNode *detectCycle(ListNode *head) {
	ListNode* slow = head, *fast = head;
	while(fast && fast->next) {
		slow = slow->next;
		fast = fast->next->next;
		if(fast == slow) {
			slow = head;
			// refer to my notes
			while(fast != slow) {
				fast = fast->next;
				slow = slow->next;
			}
			return slow;
		}
	}
	return nullptr;
}

// easy map (or set) solution
ListNode *detectCycle(ListNode *head) {
	map<ListNode*, int> m;
	while(head) {
		if(m[head] != 0) return head;
		m[head] = 1;
		head = head->next;
	}
	return nullptr;
}
```
## Stock Maximize (Hackerrank)
* Your algorithms have become so good at predicting the market that you now know what the share price of Wooden Orange Toothpicks Inc. (WOT) will be for the next number of days. Each day, you can either buy one share of WOT, sell any number of shares of WOT that you own, or not make any transaction at all. What is the maximum profit you can obtain with an optimum trading strategy? 
* Trick question!! No need for DP, simply traverse backward
```c++
long stockmax(vector<int> prices) {
	long n = prices.size(), ans=0;
	int cur_max=prices[n-1];
	for(int i=n-1; i >= 0; i--) {
		cur_max = max(prices[i], cur_max);
		ans+=(cur_max - prices[i]);
	}
	return (ans < 0? 0 : ans);
}
```
## Red John is Back (Hackerrank)
* There is a wall of size 4xn in the victim's house. The victim has an infinite supply of bricks of size 4x1 and 1x4 in her house. There is a hidden safe which can only be opened by a particular configuration of bricks. First we must calculate the total number of ways in which the bricks can be arranged so that the entire wall is covered.
* DP IDEA: there's two cases to consider
	* if the last brick (rightmost) is vertical, then we can achieve this configuration by simply adding 1 vertical brick to the previous configuration, therefore v[i-1]
	* if the last brick (rightmost) is horizontal, then we know the three bricks below it have to be horizontal. We therefore eliminate a 4 bricks, therefore v[i-4]
```c++
int redJohn(int n) {
	vector<int> v(n+1,0);
	for(int i=0; i <= n; i++) {
		if(i <= 3) v[i] = 1;
		else v[i] = v[i-1]+v[i-4];
	}
	return v[n];
}
```
## Nikita and the Game
* In each move, Nikita must partition the array into 2 non-empty contiguous parts such that the sum of the elements in the left partition is equal to the sum of the elements in the right partition. If Nikita can make such a move, she gets 1 point; otherwise, the game ends.
* After each successful move, Nikita discards either the left partition or the right partition and continues playing by using the remaining partition as array arr.
* Find the max score!
```c++
long long solve(long long start, long long end, vector<int>& arr, long long sum) {
    if(sum == 0) return end-start-1; // when sum == 0, maximal times to be split: end-start-1, take care of the case where all elements in arr are 0s
    if(sum % 2 == 1) return 0; // odd numbers can't be split
    long long half = sum / 2, s = 0;
    for(long long i=start; i < end; i++) {
        s+=arr[i];
        if(s == half) 
            return 1+max(solve(start, i+1, arr, s), solve(i+1, end, arr, s));
        else if(s > half) break;
    }
    return 0;
}

long long arraySplitting(vector<int> arr) {
    long long sum = 0;
    for(int i : arr) sum +=i;
    return solve(0, arr.size(), arr, sum);
}
```
## Longest String Chain
* You are given an array of words where each word consists of lowercase English letters. wordA is a predecessor of wordB if and only if we can insert exactly one letter anywhere in wordA without changing the order of the other characters to make it equal to wordB.
	* For example, "abc" is a predecessor of "abac", while "cba" is not a predecessor of "bcad".
* A word chain is a sequence of words [word1, word2, ..., wordk] with k >= 1, where word1 is a predecessor of word2, word2 is a predecessor of word3, and so on. A single word is trivially a word chain with k == 1.
* Return the length of the longest possible word chain with words chosen from the given list of words.
```c++
int longestStrChain(vector<string>& words) {
	sort(words.begin(), words.end(), [](std::string a, std::string b) {return a.length() < b.length(); });
	map<string, int> um;
	int ans=0;
	for(int i=0; i < words.size(); i++) {
        um[words[i]] = 1;
        for(int j=0; j < words.size(); j++) {
			string prev = words[i].substr(0,j) + words[i].substr(j+1);
			if(um.find(prev) != um.end()) {
				um[words[i]] = max(um[prev]+1, um[words[i]]);
				ans=max(um[words[i]], ans);
			}
		}
	}
	return ans;
}
```
## The Longest Common Subsequence (Hackerrank)
* Given two sequences of integers, A=[a[1]...a[n]] and B=[b[1]...b[m]], find the longest common subsequence and print it as a line of space-separated integers. If there are multiple common subsequences with the same maximum length, print any one of them.
```c++
vector<int> longestCommonSubsequence(vector<int> a, vector<int> b) {
    int n=a.size();
    int m=b.size();
    int dp[n+1][m+1];
    for(int i=0;i<=n;i++){
        for(int j=0;j<=m;j++){
            if(i==0 || j==0)
                dp[i][j]=0;
        }
    }
    for(int i=1;i<=n;i++){
        for(int j=1;j<=m;j++){
            if(a[i-1]==b[j-1])
                dp[i][j]=1+dp[i-1][j-1]; // if the same, count the current element
            else
                dp[i][j]=max(dp[i-1][j],dp[i][j-1]); // else count the max of including either element
        }
    }
    int i=n;
    int j=m;
    vector<int> v;
    // backtrack
    while(i>0 && j>0){
        if(a[i-1]==b[j-1]){
            v.push_back(a[i-1]);
            i--;
            j--;
        }
        else{
            if(dp[i-1][j]<dp[i][j-1])
                j--;
            else 
                i--;
        }
    }
    reverse(v.begin(),v.end());
    return v;

}
```
## Mandragora Forest
* As she encouters each mandragora, her choices are:
	* Garnet's pet eats mandragora **i**. This increments **s** by **1** and defeats mandragora **i**.
	* Garnet's pet battles mandragora **i**. This increases **p** by **s\*H[i]** experience points and defeats mandragora **i**.
* IDEA: gready!! sort first!! Add the smaller-valued H[i] to increment s 
```c++
long mandragora(vector<int> H) {
    sort(H.begin(),H.end());
    long sum=0;
    for(int i=0; i < H.size(); i++) {
        sum+=H[i];
    }
    
    long ans=sum;
    int s=1;
    for(int i=0; i < H.size(); i++) {
        s++;
        sum-=H[i];
        long tmp = s*sum;
        ans = max(ans, tmp);
    }
    return ans;
}
```
## Count Ways to Split
* Given a string s, task is to count the number of ways of splitting s into three non-empty parts a, b and c in such a way that a + b, b + c and c + a are all different strings.
* Note ab, bc, ca all need at least two characters!!!! --> set the for loop initialization accordingly
```
int countWaysToSplit(string s) {
    int ans = 0;
    for(int i=1; i <= s.size()-2; i++) {
        for(int j=i+1; j <= s.size()-1; j++) {
            // string a =s.substr(0,i);
            // string b =s.substr(i,j-i);
            // string c =s.substr(j);
            string ab = s.substr(0,j);
            string bc = s.substr(i);
            string ca = s.substr(j) + s.substr(0,i);
            if(ab == bc || bc == ca || ab == ca) continue;
            else ans++;
        }
    }
    return ans;
}
```
## University Career Fair
* Given companies arrival times and durations, output the max number of companies that can visit the place (only 1 allowed at a time)
* Greedy!
```c++
int maxEvents(vector<int> arrival, vector<int> duration) {
	vector<vector<int>> v;
	for(int m=0; m < arrival.size(); m++) {
		v.push_back({arrival[m]+duration[m], duration[m]});
	}
	// sort by end time!!! Sort by start time doesn't tell much information
	sort(v.begin(), v.end());
	int end=INT_MIN, ans=0, i=0;
	while(i < v.size()) {
		if(arrival[i] >= end) {
			ans++;
			end = v[i][0];
		}
		i++;
	}
	return ans;
}
```
## Play with words (Hackerrank)
* DP!!! 
* Get 2 non-overlapping palindromic subsequences from a string. The score obtained is the product of the length of these 2 subsequences. Output the max score
* https://www.geeksforgeeks.org/longest-palindromic-subsequence-dp-12/
```c++
int playWithWords(string s) {
     
     int n=s.size(),m=0;
     // dp[i][j] represents the max number of palindromic subsequences from in the substring from index i to j inclusive
     // half of the entries unused since i <= j
     vector<vector<int>> dp(n);
     for(int i=0;i<n;i++){
         dp[i]=vector<int>(n);
         dp[i][i]=1;
     }
     
     // cl equals number of letters in the string
     // i = starting position, j = ending position (both inclusive)

     for (int cl=2; cl<=n; cl++) {
        for (int i=0; i<n-cl+1; i++){
            int j = i+cl-1; 
            // if two letters and they are the same
            if (s[i] == s[j] && cl == 2) 
               dp[i][j] = 2; 
            
            // from index i+1 to j-1 plus the two outer letters
            else if (s[i] == s[j]) 
               dp[i][j] = dp[i+1][j-1] + 2; 
            else
               dp[i][j] = max(dp[i][j-1], dp[i+1][j]); 
        } 
    }
    for(int i=0;i<n-1;i++){
        if(dp[0][i]*dp[i+1][n-1] > m)
            m=dp[0][i]*dp[i+1][n-1];
    } 
    return(m);

}
```
## The Indian Job
* Given a set of integers, the task is to divide it into two sets S1 and S2 such that the absolute difference between their sums is minimum, and check if sum of S1 and sum of S2 are both less than g
```c++
string indianJob(int g, vector<int> arr) {
    int sum = 0, n = arr.size();
    for (int i = 0; i < n; i++)
        sum += arr[i];
    // dp[i] gives whether is it possible to get i as sum of elements, only need half of the sum space since the other half can be achieved by sum-i
    int y = sum / 2 + 1;
    
    // dd helps us not double count the current element
    bool dp[y], dd[y];
 
    // Initialising dp and dd
    for (int i = 0; i < y; i++) {
        dp[i] = dd[i] = false;
    }
 
    // sum = 0 is always possible
    dd[0] = true;
    for (int i = 0; i < n; i++) {
        // updating dd[k] as true if k can be formed using previous elements and current i
        // note if a single element is larger than half of the sum, it gets skipped, since its value can be achieved by (total sum - sum of other elements)
        for (int j = 0; j + arr[i] < y; j++) {
            if(dp[j])
                dd[j + arr[i]] = true;
        }
        
        // updating dd to dp
        // dd used so don't count the same value twice in one iteration
        for (int j = 0; j < y; j++) {
            if(dd[j])
                dp[j] = true;
			// reset dd
            dd[j] = false; 
        }
    }
 
	// check the biggest i only since it gives the most even 
    for (int i = y-1; i >= 0; i--) {
        if (dp[i]) {
			// two numbers formed are sum-i and i, both have to be less than g
            if(sum-i > g || i > g) return "NO";
            return "YES";
        }
    }
}
```
## Codility Stick cut challenge (VERY SMART!!!)
* There are two wooden sticks of lengths A and B respectively. Each of them can be cut into shorter sticks of Integer lengths. Our goal is to construct the largest possible square.
* Each stick can be cut into 2, 3, 4 pieces or not cut. Take the max!!!
```c++
int solution(int A, int B) {
    int a = A / 4;
    int b = min(A, B / 3);
    int c = min(A / 2, B / 2);
    int d = min(A / 3, B);
    int e = B / 4;
    return max(a, max(b, max(c, max(d, e))));
}
```
