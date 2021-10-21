#  LeetCode notes
## Longest increasing subsequence (LIS) (DP!)
```c++
int findLIS(vector<int> sequence)  {
	// store the length of the longest increasing sequence at that index
	vector<int> lis(sequence.size(), 1); 
	int ans = 0;
	for (int n = 1; n < sequence.size(); n++) {
		for (int i = 0; i < n; i++) {
			// if decreasing at any point
	        if (sequence[i] >= sequence[n]) continue;
        	lis[n] = max(lis[n], 1 + lis[i]);
        	ans = max(ans, lis[n]);
	    }
	}
	return ans;
}

---

vector<int> table;
table.push_back(sequence[0]);

for (int i = 1; i < sequence.size(); ++i) {
    int goal = sequence[i];
    if (goal > table[table.size() - 1]) {
        table.push_back(goal);
    } 
    else {
        *lower_bound(table.begin(), table.end(), goal) = goal;
    }
}
```

## Longest Common subsequence (LCS) (DP!)
* create a 2D array, fill in, return the v[size-1][size-1] number as the number of chars of the LCS
```c++
for (int testcase = 0; testcase < t; ++testcase) {

	cout << "Case " << testcase+1 << ": ";
	
	
	int n, p, q;
	cin >> n >> p >> q;
	vector<int> base(n*n+1, -1);

	for(int i=0; i < p+1; i++) {
		int num;
		cin >> num;
		base[num] = i;
	}
	
	vector<int> cleaned;
	for(int i=0; i < q+1; i++) {
		int tmp;
		cin >> tmp;
		if(base[tmp] != -1) {
			cleaned.push_back(base[tmp]);
		}
	}

	if (cleaned.size() == 0) {
		cout << "0\n";
		continue;
	}


	vector<int> lis;
	lis.push_back(cleaned[0]);

	for(int i=1; i < cleaned.size(); i++) {
		if(cleaned[i] > lis[lis.size()-1]) {
			lis.push_back(cleaned[i]);
		}
		else {
			*lower_bound(lis.begin(), lis.end(), cleaned[i]) = cleaned[i];
		}
	}

	cout << lis.size() << endl;

}
```
	
## Max profit
```c++
int maxProfit(vector<int>& prices) {
	int minn=prices[0];
	int result=0;

	for(int i=1; i < prices.size(); i++) {
    	minn = min(minn, prices[i]);
   	 	result = max(result, prices[i]-minn);
	}

	return result;

}
```

## Remove duplicates (two pointers)
```c++
//at most two consecutive numbers
int removeDuplicates(vector<int>& nums) {

    int lag=1, count=1, i=1;

    while(i < nums.size()) {
        if(nums[i] == nums[i-1]) count++;
        else count = 1;

        if(count <= 3) nums[lag++] = nums[i++];
        else i++;
	}
	return lag;
}
```

## Unique path (DP!):
```
int uniquePaths(int m, int n) {
    int arr[m][n];
    for(int i=0; i < m; i++) {
        arr[i][0] = 1;
	}

    for(int i=0; i < n; i++) {
        arr[0][i] = 1;
	}

    for(int r=1; r < m; r++) {
	    for(int c=1; c < n; c++) {
		    arr[r][c] = arr[r-1][c] + arr[r][c-1];
		}
	}

    return arr[m-1][n-1];
}
```


## Gray code:
```c++
//no binary needed
vector<int> grayCode(int n) {
	vector<int> ans;
	ans.push_back(0);
	ans.push_back(1);

	int index=2;
	for(int i=2; i <=n; i++) {
	//reflect the previous sequence
		for(int j=index-1; j >=0; j--) {
			ans.push_back(ans[j]);
		}

		//add a fix number to it
		for(int j=index; j <2*index; j++) {
			ans[j] = ans[j] + index;
		}￼
		
		index*=2;
	}
	return ans;
}
```

![image](https://user-images.githubusercontent.com/50003319/106405714-c8667b80-63fc-11eb-8a24-98d53e814f1b.png)


## Combination type problem:
```c++
void resCombination(int n, int k, vector<int> curr, int index){
//	if(index > n) return;
	if(curr.size() == k) {
		v.push_back(curr);
		return;
	}

	for(int i=index; i < n; i++) {
		curr.push_back(i);
		resCombination(n, k, curr, i+1);
		curr.pop_back();
	}
	return;
}
```
￼￼
## Reverse tree level order traversal
```c++
vector<vector<int>> levelOrderBottom(TreeNode* root) {
	vector<vector<int>>ans;

	queue<TreeNode*> queue;
	if(root==NULL)
	return ans;
	queue.push(root);
	ans.push_back({root->val});

	while(!queue.empty()) {
		int size=queue.size();
		vector<int>temp;
		
		while(size) {
			TreeNode* num=queue.front();
			queue.pop();

			if(num->left) {	
				queue.push(num->left);
				temp.push_back(num->left->val);
			}
			if(num->right) {
				queue.push(num->right);
				temp.push_back(num->right->val);
			}
			size--;
		}
		
		ans.push_back(temp);

	}

	ans.pop_back();
	reverse(ans.begin(),ans.end());
	return ans;
}
```

* Traverse creates in-order traversal of the tree
* Store it in m_q
```c++
class BSTIterator {
public:
	BSTIterator(TreeNode* root) {
		traverse(root);
	}
	
	int next() {
		int val = m_q.front();
		m_q.pop();
		return val;
	}
	
	bool hasNext() {
		return !m_q.empty();
	}
private:
	void traverse(TreeNode* root) {
		if(!root) return;
		//traverse left branches first, then middle, then right
		traverse(root->left);
		m_q.push(root->val);
		traverse(root->right);
	}
	queue<int> m_q;
};
```
## Container With Most Water
```c++
int maxArea(vector<int>& height) {
	int lo = 0;
	int hi = height.size()-1;
	int maxarea = INT_MIN;
	while(lo < hi){
		int temp = min(height[lo],height[hi])*(hi-lo);
		maxarea = max(maxarea,temp);
		if(height[hi]>height[lo])
			lo++;
		else
			hi--;
	}

	return maxarea;
}
```
## Discussion question
* For each position i (from 0 to n-1) of sequence, this function should find the smallest index j such that j > i and sequence[j] > sequence[i], and put sequence[j] in results[i]; if there is no such j, put -1 in sequence[i]
```c++
void findNextInts(const int sequence[], int results[], int n) {
	
	if(n <= 0) return;
	stack<int> s;
	s.push(0);
	
	for(int i=1; i < n; i++) {
		int curr = sequence[i];
		if(!s.empty() && curr > sequence[s.top()]) {
			results[s.top()] = curr;
			s.pop();
		}
		s.push(i);
	}
	
	while(!s.empty()) {
		results[s.top()] = -1;
		s.pop();
	}
}
```
## Minimum Path Sum (DP!)
* note the initialization of 2D vector -- the row number comes first, followed by a vector containing the column number. if want initial value with vector, put it in the 1D vector
```c++
int minPathSum1(vector<vector<int>>& grid) {
	
	vector<vector<int>> dp (grid.size(), vector<int>(grid[0].size()));
	dp[0][0] = grid[0][0];
	
	for(int i=1; i < grid[0].size(); i++) {
		dp[0][i] = grid[0][i] + dp[0][i-1];
	}
	
	for(int i=1; i < grid.size(); i++) {
		dp[i][0] = grid[i][0] + dp[i-1][0];
	}
	
	for(int i=1; i < grid.size(); i++) {
		for(int j=1; j < grid[0].size(); j++) {
			dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1]);
		}
	}
	return dp[grid.size()-1][grid[0].size()-1];
}
```

## Permutations
```c++
vector<vector<int>> ans;
void find(vector<int>& nums, int l) {
	int r = nums.size()-1;
	if(l == r) ans.push_back(nums);
	else {
		for(int i=l; i <= r; i++) {
			swap(nums[i], nums[l]);
			find(nums,l+1);
			swap(nums[i], nums[l]);
		}
	}
}

vector<vector<int>> permute(vector<int>& nums) {
	find(nums, 0);
	return ans;
}
```

## Permutations II
```c++

vector<vector<int>> anss;
//initialize to false by default
vector<bool> used;
void backtrack(vector<int> nums,vector<int> temp) {
	if(temp.size()==nums.size()) {
		anss.push_back(temp);
		return;
	}

	//always adding the first element here
	for(int i=0;i<nums.size();i++) {
		//skip because will generate same permu
		if(!used[i]) {
			temp.push_back(nums[i]);
			used[i]=true;
			backtrack(nums,temp);
			temp.pop_back();
			used[i]=false;
			while(i < nums.size()-1 && nums[i] == nums[i+1]) i++;
		}

	}
}

vector<vector<int>> permuteUnique(vector<int>& nums) {
	int n=nums.size();
	used.resize(n);
	vector<int> temp;
	sort(nums.begin(), nums.end());
	backtrack(nums,temp);
	return anss;
}
```
## Maximum Depth of Binary Tree
* maximum number of nodes
```c++
int maxDepth(TreeNode* root) {
	if(root == nullptr) return 0;
	return max(maxDepth(root->left)+1, maxDepth(root->right)+1);
}
```

## Single Number II
* bit manipulation(!?)
* find the num that appears only once (others thrice)
```c++
int singleNumber(vector<int>& nums) {
	int i=0;
	int j=0;
	for(int k:nums){
		i=~j&(i^k);
		j=~i&(j^k);
	}
	return i ;  
}
```

## Linked List Cycle 
* use a faster pointer and a slower pointer, if the faster catches up the slower, then there's a loop
* can use assignment directly in the if statement
```c++
bool hasCycle(ListNode *head) {
	ListNode* p=head;
	while(p && p->next) {
		if((p=p->next->next) == (head=head->next)) return true;
	}
	return false;
}
```


## Combination Sum
* solve it myself!!!!
```c++
vector<vector<int>> ans;
void solve(vector<int>& candidates, vector<int>& v, int target, int index) {
	if(target == 0) {
		ans.push_back(v);
		return;
	} 
	if(target < 0) return;
	
	//have to use index, so don't reuse the previous values and get something like 2, 3, 2
	for(int i=index; i < candidates.size(); i++) {
		v.push_back(candidates[i]);
		solve(candidates, v, target-candidates[i], i);
		v.pop_back();
	}
}

vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
	sort(candidates.begin(), candidates.end());
	vector<int> v;
	solve(candidates, v, target, 0);
	return ans;
}
```

## Combination Sum II
* fixed!!
```c++
vector<vector<int>> ans;
void solve(vector<int>& candidates, vector<int>& v, int target, int index) {
	if(target == 0) {
		ans.push_back(v);
		return;
	} 
	if(target < 0) return;

	for(int i=index; i < candidates.size(); i++) {
		v.push_back(candidates[i]);
		//index = i+1 because don't readd the current value
		solve(candidates, v, target-candidates[i], i+1);
		v.pop_back();
		//skipping the duplicate values
		//can't skip in the beginning of the loop because cases like 1,1,6 would be excluded
		//don't want duplicate sets not duplicate values in a set! 
		while(i+1 < candidates.size() && candidates[i] == candidates[i+1]) i++;
	}
}

vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
	sort(candidates.begin(), candidates.end());
	vector<int> v;
	solve(candidates, v, target, 0);
	return ans;
}
```
## Subsets II
* similar concept as combination
```c++
vector<vector<int>> ans;
void solve(vector<int>& nums, vector<int>& v, int index) {
	ans.push_back(v);
	if(index==nums.size()) return;
	for(int i=index; i < nums.size(); i++) {
		v.push_back(nums[i]);
		solve(nums, v, i+1);
		v.pop_back();  
		while(i+1 < nums.size() && nums[i] == nums[i+1]) i++;
	}
}
vector<vector<int>> subsetsWithDup(vector<int>& nums) {
	sort(nums.begin(), nums.end());
	vector<int> v;
	solve(nums, v, 0);
	return ans;
}
```
## Longest Common Subsequence
* not necessarily contiguous
```c++
string longestCommonSubsequence(string s1, string s2) {
	if(s1.empty() || s2.empty()) return "";

	char firstS1 = s1[0];
	char firstS2 = s2[0];
	if(firstS1 == firstS2) return firstS2 + longestCommonSubsequence(s1.substr(1), s2.substr(1));

	//check both possibilities 
	string tmp1 = longestCommonSubsequence(s1, s2.substr(1));
	string tmp2 = longestCommonSubsequence(s1.substr(1), s2);

	return (tmp1.size() > tmp2.size() ? tmp1 : tmp2);
}
```
## Merge Sorted LinkedList (recursion)
* tricky!!
```c++
Node* merge(Node* l1, Node* l2) {
	if(l1 == nullptr) return l2;
	if(l2 == nullptr) return l1;

	Node* head;
	if(l1->val < l2->val) {
		head = l1;
		head->next = merge(l1->next, l2);
	}
	else {
		head = l2;
		head->next = merge(l1, l2->next);
	}

	return head;
}
```
## Reverse a doubly LinkedList (recursion)
* very tricky!!
```c++
Node* reverse(Node* head) {
	//if an empty list
	if(head == nullptr) return nullptr;
	
	//swap next and prev
	Node* tmp = head->next;
	head->next = head->prev;
	head->prev = tmp;
	
	//originally head->next, if the now prev is null we're done
	if(head->prev == nullptr) return head;
	return reverse(head->prev);
}
```
## Unique Paths II
* dp!
* Don't try to be smart, create a new 2D vector is easier
```c++
int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
	int m = obstacleGrid.size(), n = obstacleGrid[0].size();

	if(obstacleGrid[0][0]==1 || obstacleGrid[m-1][n-1]==1)//if start or end has obstacle 0 ways
	return 0;

	vector<vector<int>> mem(m,vector<int>(n,0));


	//fill the first column as 1 until encounter any obstacle
	for(int i=0; i<m; ++i)	{
		if(obstacleGrid[i][0] == 0)
				mem[i][0] = 1;
		else
			break;//if encounter with first obstacle leave all as 0
	}

	//fill the first row as 1 until encounter any obstacle
	for(int j=0; j<n; ++j) {
		if(obstacleGrid[0][j] == 0)
			mem[0][j] = 1;
			else
			break;//if encounter with first obstacle leave all as 0
	}
	
	for(int i=1; i<m; ++i) {
		for(int j=1; j<n; ++j) {
			if(obstacleGrid[i][j] ==1 )
				mem[i][j]= 0;
			else
				mem[i][j]= mem[i-1][j]+mem[i][j-1];
			}
	}
	return mem[m-1][n-1];
}
```
## Partition List
```c++
ListNode* partition(ListNode* head, int x) {
	ListNode* tmp = new ListNode();
	tmp->val = -10000;
	tmp->next = head;
	ListNode *c, *p, *bef = tmp;
	
	while(bef->next != nullptr && bef->next->val < x)
		bef = bef->next;
		
	c = bef->next;
	p = bef;
	
	while(c != NULL) {
		if(c->val < x) {
			p->next = c->next;
			c->next = bef->next;
			bef->next = c;
			bef = c;
			c = p->next;
		}
		
		else {
			p = c;
			c = c->next;
		}
	}

	return tmp->next;
}
```
## EndX
* move all x's in a string to the end using recursion
* substr useful!!
```c++
string endX(string str) {
	if(str.size() <= 1) return str;
	if(str[0] == 'x') return endX(str.substr(1)) + 'x';
	else return str[0] + endX(str.substr(1));
}

//original solution (less elegant)
	if(str.size() == 1) return str;
	int count=0;
	while(str[0] == 'x') {
		count++;
		int i=0;
		for(;i < str.size()-1; i++) {
			str[i] = str[i+1];
		}
		str[i] = 'x';
		if(count==str.size()) break;
	}
	return str[0] + endX(str.substr(1));
	}
```

## isSolvable
* can be written in the form of ax + by = c
```c++
bool isSolvable(int x, int y, int c) {
	if(c == 0) return true;
	if(c < 0) return false;
	return isSolvable(x, y, c-y) || isSolvable(x, y, c-x);
}
```
## LinkedList with recursion!!!
```c++
// inserts a value in a sorted linked list of integers
// returns list head
// before: 1 → 3 → 5 → 7 → 15
// insertInOrder(head, 8);
// after: 1 → 3 → 5 → 7 → 8 → 15
Node* insertInOrder(Node* head, int value) {
	if(head == nullptr || value < head->data) {
		Node* p = new Node;
		p->data = value;
		p->next = head;
		head = p;
	}
	else head->next = insertInOrder(head->next, value);
	return head;
}

// deletes all nodes whose keys/data == value, returns list head
// use the delete keyword
// @b tricky!!!!
Node* deleteAll(Node* head, int value) {
	if(!head) return nullptr;
	else {
		if(head->data == value) {
			Node* tmp = head->next;
			delete head;
			return deleteAll(tmp, value);
			}
		else {
			head->next = deleteAll(head->next, value);
			return head;
		}
	}
}


//print in reverse order
void reversePrint(Node* head) {
	if(head == nullptr) return;
	reversePrint(head->next);
	cout << head->data;
}
```
## Complex logics of STL/iterator
```c++
int deleteOddSumLists(set<list<int>*>& s) {
	int count = 0;
	set<list<int>*>::iterator it = s.begin();
	for(; it != s.end(); ) {
		int sum = 0;
		//begin() and end() can be accessed using arrows!!
		for(list<int>::iterator p=(*it)->begin(); p != (*it)->end(); p++) {
			sum += *p;
		}
		if(sum%2 == 1) {
			delete (*it);
			it = s.erase(it);
			count++;
		}
		else it++;
	}
	return count;
}
```

## Sort in O(N)!?
* use the internal order of indices of an array
```c++
void sort(int a[], int n) {
	int countTable[100] = {0};
	for(int i=0; i < n; i++) {
		countTable[a[i]-1]++;
	}

	int j =0;
	//traverse the countTable
	for(int i=0; i < 100; i++) {
		//traverse each element of countTable
		for(;countTable[i] > 0; countTable[i]--) {
			//add a number for how many times
			a[j++] = i+1;
		}
	}
}
```
## reverseBits
* '&' compares each bits and (101)&(011) = 1
```c++
uint32_t reverseBits(uint32_t n) {
	uint32_t mask=1;
	string ans;

	for(int i=0;i<32;i++){

		if((mask&n)!=0) s+="1";
		else ans+="0";

		//shift left one bit
		mask <<= 1;
	}   

	return stoll(ans,0,2);
}
```
* can return {1,2} as a vector return value!

## Unique Binary Search Trees
* DP!
```c++
int dp[30];
int numTrees(int n) {
	//1 way to arrange 1 or 0 node
	dp[0] = dp[1] = 1;
	//2 ways to arrange two nodes with values 1, 2
	dp[2] = 2;
	for(int i=3; i <= n; i++) { 
		// i represents the number of nodes
		for(int j = 0; j < i; j++) { 
			// number of combincations of left subtree times the number of combincations of right sub tree
			// if n == 3, 2 + 1 + 2 = 5
			// 2 (left 0 node 1 possibility * right 2 nodes 2 posibilities) + 1 (left 1 node 1 possibiltiy * right 1 node 1 possibility) + 2
			dp[i] += dp[j] * dp[i-j-1];
		}
	}
	return dp[n];
}
```

## Palindrome Partitioning
* more understandable approach but less efficient
* add another DP solution
```c++
bool check(string s) {
	bool x=true;
	int n=s.size();
	for(int i=0;i<=(n/2)-1;i++) {
		if(s[i]!=s[n-1-i])
			x=false;
	}
	return x;
}

void fun(string s,int i,int n,vector<string>&v,vector<vector<string>>&res) {
	if(i==n) {
		res.push_back(v);
		return;
	}
	
	// i and j keep track of the start and end indices for each substring
	for(int k=i;k<n;k++) {
		//second param represents # of characters, therefore +1
		string sp=s.substr(i,k-i+1);
		if(check(sp)) {
			v.push_back(sp);
			//change the starting index
			fun(s,k+1,n,v,res);
			v.pop_back();
		}
	}
	
	//alternatively
	/* 
	// j can be equal size to account for the last char in the string
	for(int j=i+1; j <= size; j++) {
		string sub = s.substr(i, j-i);
		if(isPalin(sub)) {
			v.push_back(sub);
			traverse(j, size, s, v, ans);
			v.pop_back();
		}
	}
	*/
}

vector<vector<string>> partition(string s) {
	vector<vector<string>>res;
	vector<string>v;
	int n=s.size();
	fun(s,0,n,v,res);
	return res;  
}
```
## My first graph problem!!!!
* pay attention to conversion between vectors, chars and ints
```c++
vector<vector<int>> graph(26);

void dfs(int a, vector<bool>& b) {
	b[a] = true;
	//has to start at 0!! a is the index not the first element
	for(int i=0; i < graph[a].size(); i++) {
		if(!b[graph[a][i]])
			dfs(graph[a][i], b);
	}
}

string solve(const string& s1, const string& s2) {
	if(s1.size() != s2.size()) return "no";
	for(int i=0; i < s1.size(); i++) {
		//each char should have its own b
		vector<bool> b(26);
		dfs(s1[i] - 'a', b);
		//if can't find a match at the corresponding letter
		if(b[s2[i] - 'a'] == false) return "no";
	}
	return "yes";
}

int main()
{
	int n, m;
	cin >> n >> m;
	while(n--) {
		char first, second;
		cin >> first >> second;
		graph[first-'a'].push_back(second-'a');
	}

	while(m--) {
		string s1, s2;
		cin >> s1 >> s2;
		cout << solve(s1, s2) << endl;
	}

}
```
## First time using deque!
* improved mazing-solving
```c++
bool SolveMaze(vector<vector<int>> grid, int startX, int startY, int endX, int endY) {
int n = grid.size();
int m = (n == 0 ? 0 : grid[0].size());
if(m == 0 || n == 0) return false;

if(startX == endX && startY == endY) return true;

//good trick! makes codes cleaner later on, don't have to write a bunch of if statements
int xd[] = {1, 0, -1, 0};
int yd[] = {0, 1, 0, -1};

//use pair instead of making our own struct like Coordinates
deque<pair<int, int>> dq;
dq.push_back(make_pair(startX, startY));

while(!dq.empty()) {
pair<int, int> coords = dq.front();
dq.pop_front();

int x = coords.first;
int y = coords.second;

//marked as visited
grid[x][y] = -1;
if(x == endX && y == endY) return true;

//push back neighboring blocks
for(int i=0; i < 4; i++) {
int xa = x + xd[i];
int ya = y + yd[i];

//if out of bound, visited, is wall
if(xa < 0 || xa >= n || ya < 0 || ya >= m || grid[xa][ya] == -1 || grid[xa][ya] == 1)
continue;

dq.push_back(make_pair(xa, ya));
}
}

return false;
}
```
## Firetrucks are red
* very very trickyyy!
* unodered map + graph + bfs
```c++
int main()
{
	int n; cin>>n;
	vector<vector<pair<int, int>>> adj_list(n+1);
	unordered_map<int, int> map;

	for(int i=1; i <= n; i++) {
		int m; cin >> m;
		while(m--) {
			int d; cin >> d;
			if(map.find(d) == map.end()) {
				//assign the position to the weight
				map[d] = i;
			}

			else {
				//do it twice so it's undirected
				adj_list[i].push_back(make_pair(d, map[d]));
				//d is the weight
				adj_list[map[d]].push_back(make_pair(d, i));
			}
		}
	}

	queue<int> q;
	vector<bool> visited(n+1);
	vector<int> output; //size should be the number of edges

	int vertex = 1;
	visited[vertex] = true;
	q.push(vertex);

	//running bfs
	while(!q.empty()) {
		vertex = q.front();
		q.pop();
		for(int i=0; i < adj_list[vertex].size(); i++) {
			int adjacent = adj_list[vertex][i].second;
			int weight = adj_list[vertex][i].first;
			if(!visited[adjacent]) {
			//set to true because don't want duplicates, want to build a spanning tree, should be undirected
				visited[adjacent] = true;
				//vertex and adjacent therefore share the same weight
				output.push_back(vertex);
				output.push_back(adjacent);
				output.push_back(weight);
				//continue check the adj_list at location [adjacent]
				q.push(adjacent);
			}
		}
	}

	//check if the outputs are valid, (a tree with n vertices have n-1 edges)
	if(output.size() != 3*(n-1)) {
		cout << "impossible" << endl;
	}
	else {
		for(int i=0; i < output.size(); i+=3) {
			cout << output[i] << " " << output[i+1] << " " << output[i+2] << endl;
		}
	}
}
```

## Lost Map
* Prim's algorithm --> to find the minimal spanning tree
![prim's algo psuedo](https://user-images.githubusercontent.com/50003319/109755029-d00b7280-7baa-11eb-9de2-029e3806e7ae.jpg)
![prim's algo code](https://user-images.githubusercontent.com/50003319/109755024-cda91880-7baa-11eb-80c8-371c62b0fafd.jpg)

* tuple!? keep track of multiple numbers, note the syntax
```c++
//trying to find the minimal spanning tree
//use Prim's algorithm
int main() {
	int n;
	cin >> n;
	vector<vector<int>> graph(n, vector<int>(n));
	//keep track of if visited, want undirected tree
	vector<bool> visited(n, false);

	//build a adjacency matrix
	for(int i=0; i < n; i++) {
		for(int j=0; j < n; j++) {
		cin >> graph[i][j];
		}
	}

	//cool af!!!
	//need to keep trakc of the weight, current node, and the node it came from
	typedef tuple<int, int, int> triple;
	//<data type, container to use, priority type>
	//sort the first element of the tuple (aka the weight) in increasing order --> same applies to pair
	priority_queue<triple, vector<triple>, greater<triple>> edges;
	edges.push({0,0,0});
	
	while(!edges.empty()) {
		triple e = edges.top();
		edges.pop();
		//weight, vertex we came from (parent), current vertex
		int w = get<0>(e), p = get<1>(e), c = get<2>(e);
		if(!visited[c]) {
			visited[c] = true;
			//if statement ignores the first iteration, +1 bc of indexing
			//should print n-1 edges
			if(c != 0) cout << p+1 << " " << c+1 << "\n";
			for(int k=0; k < n; k++) {
				//push new adjacent weight to the current,  c becomes parent, each adjacent vertex becomes new current
				//if statement unnecessary but reduce number of checks
				if(!visited[k]) edges.push({graph[c][k], c, k});
			}
		}
	}

}
```
## Tri tiling (DP!)
* number of ways to tile a 3 x n rectangle with 2 × 1 dominoes
* try to understand it!!
```c++
int main() {
	vector<int> table(31);
	table[0] = 1;
	table[1] = 2;
	table[2] = 3;

	for(int i=3; i <= 30; i++) {
		if(i%2 == 0)
			table[i] = table[2] * table[i-2] + table[i-3];
		else
			//still compute if it's odd number (treat it as missing a square) to facilate calculation
			table[i] = table[i-2] + 2*table[i-1];
	}

	int n;
	cin >> n;
	while(n != -1) {
		if(n%2 == 0) cout << table[n] << endl;
		else cout << 0 << endl;
		cin >> n;
	}
}
```
