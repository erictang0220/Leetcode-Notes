#  More codes
* general tips
* cast an overflowing int to long--> need to cast two times: long l = (long) n*n
## Knapsack (DP!)
* review!
```c++
int main()
{
	//C is the capacity, n is the # of objects
	int C, n;
	while(cin >> C && cin >> n) {
		vector<int> values(n), weight(n);
		for(int i=0; i < n; i++) {
			cin >> values[i] >> weight[i];
		}
	
		//DP array: (n+1) x (C+1)
		vector<vector<int>> B(n+1, vector<int>(C+1, 0));
		for(int k=1; k <= n; k++) {
			for(int c=1; c <= C; c++) {
				//if current object can't fit
				if(weight[k-1] > c)
					B[k][c] = B[k-1][c];
				else
					//values and weight are one index behind!! They start at 0
					//refer to https://www.youtube.com/watch?v=rPq0p9E8BN0 to review
					B[k][c] = max(B[k-1][c], B[k-1][c-weight[k-1]] + values[k-1]);
			}
		}

		//backtracking: to get the indices that constitute the max value
		vector<int> indices;
		int best = B[n][C], c = C;
		//if best == 0, then we know we counted all
		for(int k=n; k > 0 && best > 0; k--) {
			//if the lower row doesn't contain the same value, then we know the current index was counted
			if(B[k-1][c] != best) {
				indices.push_back(k-1);
				best -= values[k-1];
				//c decreased by the current weight
				c -= weight[k-1];
			}
		}

		cout << indices.size() << endl;
		//print in reverse order
		for(int i=indices.size()-1; i >= 0; i--) {
			cout << indices[i] << " ";
		}
		cout << endl;
		
	}

}
```

## isSubtree
* two functions required
* check if potential is a subtree of main
```c++
// check if the two trees based at r1 and r2 are identical
bool identical(Node* r1, Node* r2) {
	if(r1 == nullptr && r2 == nullptr) return true;
	if(r1 == nullptr || r2 == nullptr) return false;

	return (r1->val == r2->val && identical(r1->left, r2->left) && identical(r1->right, r2->right));
}

bool isSubtree(Node* main, Node* potential) {

	//empty tree is always a subtree
	if(potential == nullptr) return true;

	// an empty tree has no subtree
	if(main == nullptr) return false;

	if(identical(main, potential)) return true;

	// truncate main to see if its substree equal potential
	return isSubtree(main->left, potential) || isSubtree(main->right, potential);
}
```

## Print level-order tree
```c++
void printLevelOrder(Node* root)
{	
	//can get height from esay recursion
	int h = height(root);
	int i;
	for (int i = 1; i <= h; i++) {
		printGivenLevel(root, i);
		cout << endl;
	}
}

// print all nodes in the given level of the tree! t is the root
void printGivenLevel(Node* t, int level) {
	if(t == nullptr) return;
	if(level == 1) cout << t->val;
	else if(level > 1){
		printGivenLevel(t->left, level-1);
		printGivenLevel(t->right, level-1);
	}
}
```
```c++
// iterative method
// use nodeCount to help print line breaks!!
void levelOrder(Node* root) {
	if(root == nullptr) return;
	queue<Node*> q;
	q.push(root);

	int i=0, k=0;
	while(!q.empty()) {
		int nodeCount = q.size();
		while(nodeCount--) {
			Node* tmp = q.front();
			q.pop();
			if(tmp->left) q.push(tmp->left);
			if(tmp->right) q.push(tmp->right);
			cout << tmp->val;
		}
		cout << endl;
	}
}
```
## Group anagrams using hash map
```c++
vector<vector<string>> groupAnagrams(vector<string> strs) {
	unordered_map<int, vector<string>> words;
	for(int i=0; i < strs.size(); i++) {
		words[calculateHash(strs[i])].push_back(strs[i]);
	}

	vector<vector<string>> ret;
	unordered_map<int, vector<string>>::iterator it = words.begin();
	for(; it != words.end(); it++) {
		ret.push_back(it->second);
	}

	return ret;
}
```
## isMinHeap
* only return true for leaf nodes
```c++
bool isMinHeap(Node* head){
	// all true cases are taken care here!
	if(head == nullptr) return true;
	// take care of false cases
	if(head->left && head->left->val < head->val) return false;
	if(head->right && head->right->val < head->val) return false;
	return isMinHeap(head->left) && isMinHeap(head->right);
}
```
## Copying a list with a random node(?
* the random node points to a random node in the list, can be itself
* use hash map to keep track of what was created
```c++
struct Node {
	Node(int n) {
		data = n;
	}
	int data;
	Node* next;
	Node* random;
};

unordered_map<Node*, Node*> track;

Node* copyRandomList(Node* head) {
	if(head == nullptr) return nullptr;
	if(track.find(head) != track.end()) {
		return track[head];
	}
	Node* root = new Node(head->data);
	root->next = copyRandomList(head->next);
	root->random = copyRandomList(head->random);
	return root;
}
```
## Bachet's game
* DP!
* start out with num cards, each player can remove a number of cards in moves
```c++
int main() {
	int n;
	while(cin >> n) {
		int num;
		cin >> num;
		vector<int> moves;
		while(num--) {
			int m;
			cin >> m;
			moves.push_back(m);
		}

		vector<bool> states(n+1);
		states[0] = false;
		for(int i=1; i <= n; i++) {
			bool isWinning = false;
			for(int m : moves) {
				//only has to satisfy one of the previous losing conditions to win
				if(i - m >= 0 && states[i-m] == false) {
					isWinning = true;
					break;
				}
			}
			states[i] = isWinning;
		}

		if(states[n] == true) cout << "Stan wins" << endl;
		else cout << "Ollie wins" << endl;
	}
}
```

## Merge intervals
* no need for recursion
* STL sorting 2D vector based on the first-column number
```c++
vector<vector<int>> merge(vector<vector<int>>& intervals) {
	sort(intervals.begin(), intervals.end());
	vector<vector<int>> ret;
	int i=0;
	while(i < intervals.size()) {
		int k = i+1;
		int e = intervals[i][1];
		while(k < intervals.size() && e >= intervals[k][0]) {
			e = max(e, intervals[k++][1]);
		}
		ret.push_back({intervals[i][0], e});
		i = k;
	}
	return ret;
}
```
## Air conditioned minions!!!
* sort based on the upper bound of each interval, if same then based on lower bound
```c++
bool cmp(pair<int, int> lhs, pair<int, int> rhs) {
	if(lhs.second == rhs.second) {
		return lhs.first < rhs.first;
	}
	return lhs.second < rhs.second;
}

int main() {
	int n;
	cin >> n;
	vector<pair<int, int>> minions;

	for(int i = 0; i < n; i++) {
		pair<int, int> p;
		cin >> p.first >> p.second;
		minions.push_back(p);
	}

	// sort based on second number first
	sort(minions.begin(), minions.end(), cmp);

	int room = 1;
	int threshold = minions[0].second;

	for(int i=1; i < minions.size(); i++) {
		if(minions[i].first > threshold) {
			threshold = minions[i].second;
			room++;
		}
	}

	cout << room << endl;

}
```
## A1 paper
* interesting question~
```c++
#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

typedef long long ll;
typedef long double ld;

int main() {

	ll n;
	cin >> n;
	n--;

	vector<ll> v(n);
	for(auto& i : v) {
	cin >> i;
	}

	ld totalcost = 0;
	ll needed = 1;
	bool enough = false;

	ld longedge  = pow(2.0, -3/4.0);
	ld shortedge = pow(2.0, -5/4.0);

	for(ll i = 0; i < v.size(); i++) {

		// Calculate cost here
		totalcost += needed * longedge;
		swap(shortedge, longedge);
		// original longedge
		shortedge /= 2;

		// Calculate paper needed now
		needed *= 2;
		needed -= v[i];

		// Check if we have enough paper
		if(needed <= 0) {
			enough = true;
			break;
		}
	}

	cout << fixed;
	cout.precision(9);

	if(enough) {
		cout << totalcost << endl;
	}
	else {
		cout << "impossible" << endl;
	}
}
```
## permutations of strings (hackerRank)
* find the next permutation in lexicographically increasing order, return 0 if none found
* tricky!! understand more
```c++
int next_permutation(int n, char **s)
{
	/**
	* Complete this method
	* Return 0 when there is no next permutation and 1 otherwise
	* Modify array s to its next permutation
	*/
	// Find non-increasing suffix
	int i = n-1;
	while(i>0 && strcmp(s[i-1],s[i])>=0) 
		i--;    // find key
	if (i<=0) return 0;

	// Swap key with its successor in suffix
	int j = n-1;
	while(strcmp(s[i-1],s[j])>=0) 
		j--;    // find rightmost successor to key
	char *tmp = s[i-1];
	s[i-1] = s[j];
	s[j] = tmp;

	// Reverse the suffix
	j = n-1;
	while(i<j) {
		tmp = s[i];
		s[i] = s[j];
		s[j] = tmp;
		i++;
		j--;
	}
	return 1;

}
```
## print in radial numbers
<pre>
5 5 5 5 5 5 5 5 5 
5 4 4 4 4 4 4 4 5 
5 4 3 3 3 3 3 4 5 
5 4 3 2 2 2 3 4 5 
5 4 3 2 1 2 3 4 5 
5 4 3 2 2 2 3 4 5 
5 4 3 3 3 3 3 4 5 
5 4 4 4 4 4 4 4 5 
5 5 5 5 5 5 5 5 5 
</pre>

* idea: min2 changes for every print, get min compared with min1
```c++
int main() {
	int n;
	cin >> n;
	int len = 2*n-1;
	int min1,min2,min;
	// Complete the code to print the pattern.
	// for rows
	for (int i=1; i <=len; i++) {
		// for col
		for (int j=1; j<=len; j++) {
			// min diff btn vertical sides
			min1 = i<=len-i ? i -1 : len-i;
			// min diff btn horizontal sides
			min2 = j<=len-j ? j -1: len-j;
			// min diff btn vertical & horizontal sides
			min = min1<=min2 ? min1 : min2;
			// print
			printf("%d ",n-min);
		}
		printf("\n");
	}
}
```

## Sherlock and The Beast
- A Decent Number has the following properties:
1. Its digits can only be 3's and/or 5's.
2. The number of 3's it contains is divisible by 5.
3. The number of 5's it contains is divisible by 3.
4. It is the largest such number for its length.
```c++
void decentNumber(int n) {
	int y=n;
	int z=y;
	// if divisible by 3, then stop to retain the most 5's to maximize the number
	while(z%3 != 0) {
		z-=5;
	} 
	if(z < 0) {
		cout << -1 << endl;
	}
	else {
		string fives(z, '5');
		string threes(y-z, '3');
		cout << fives + threes << endl;
	}
}
```
## Spiral Matrix II
* create 2 vars to track
* a while loop with four for loops, each for a side
```c++
vector<vector<int>> generateMatrix(int n) {
	int l=0, r=n-1, k=1;
	vector<vector<int>> v(n, vector<int>(n));

	while(k < n*n) {
		for(int i=l; i <= r; i++) {
			v[l][i] = k++;
		}
		for(int i=l+1; i <= r; i++) {
			v[i][r] = k++;
		}
		for(int i=r-1; i >= l; i--) {
			v[r][i] = k++;
		}
		for(int i=r-1; i > l; i--) {
			v[i][l] = k++;
		}
		
		l++;
		r--;
	}

	return v;
}
```
## Fair Rations (Hackerrank)
* trick: 
	* if odd number of odd numbers --> no
	* else return 2*(sum of distance between two closest odd numbers) (try out with ex like 4, 9, 5, 6)
```c++
string fairRations(vector<int> B) {

	vector<int> v;
	for(int i=0; i < B.size(); i++) {
		if(B[i] % 2 == 1) v.push_back(i);
	}

	if(v.size() % 2 == 1) return "NO";
	int sum=0;
	for(int i=0; i < v.size(); i+=2) {
		sum+=2*(v[i+1]-v[i]);
	}
	return to_string(sum);

}
```
## Bomb lab phase_5 func_rec
* crazy recursion!
```c++
int func4(int edi, int esi, int edx) {
	int ebx = esi + ((edx - esi) >> 1);

	if(ebx == edi) {
		return ebx;
	}
	else if(ebx > edi) {
		int eax = func4(edi, esi, ebx - 1);
		return ebx + eax;
	}
	else {
		int eax = func4(edi, ebx + 1, edx);
		return eax + ebx;
	}
	return 0;
}
```
## Trapping Rain Water
```c++
int trap(vector<int>& height) {
	vector<int>left(height.size(),0);
	vector<int>right(height.size(),0);
	int maxim=INT_MIN;
	int i;
	
	// get the maximal height scanning from the left
	for(i=0;i<height.size();i++) {
		maxim=max(maxim,height[i]);
		left[i]=maxim;
	}
	
	maxim=INT_MIN;
	// get the maximal height scanning from the right
	for(i=height.size()-1;i>=0;i--) {
		maxim=max(maxim,height[i]);
		right[i]=maxim;
	}
	
	int sum=0;
	// the min of left[i] and right[i] is the height of water and building combined
	// find the miminum of both scans, subtract the original height to get the water trapped
	for(i=0;i<height.size();i++) {
		sum+= (min(left[i],right[i])-height[i]);
	}
	return sum;
}
```
## Partition List
* Given the head of a linked list and a value x, partition it such that all nodes less than x come before nodes greater than or equal to x.
* You should preserve the original relative order of the nodes in each of the two partitions.
* Annoying edge cases ugh
* Two main parts
	* finding the first node with node->val < x, can be the dummy node we created
	* attaching all the elements with node->val < x (if any) right after that.
```c++
ListNode* partition(ListNode* head, int x) {
	// res points to the head to avoid edge cases
	ListNode *res = new ListNode(0, head), *lastLess = NULL, *nextNode;

	// offset head to the dummy node
	head = res;
	while (head && head->next) {
		// storing nextNode
		nextNode = head->next;
		
		// finding lastLess!! Smart, take care of the case where the first node's val >= x
		if (!lastLess && nextNode->val >= x) {
			lastLess = head;
		}
		// moving elements with ->val < x after lastLess
		else if (lastLess && nextNode->val < x) {
			// removing nextNode from its original position
			head->next = nextNode->next;
			// splicing it right after lastLess and updating lastLess
			nextNode->next = lastLess->next;
			lastLess->next = nextNode;
			lastLess = nextNode;
			// current nextNode changes, don't increment
			continue;
		}
		// advancing head
		head = head->next;
	}
	return res->next;
}
```
## Unique Binary Search Trees II
* tricky recursion but trust the process!!!
* test out a case where start = 1, end = 2, i = 1 to better conceptualize
```c++
vector<TreeNode*> generateTrees(int n) {
	if(n == 0) return {nullptr};
	return recur(1, n);
}

vector<TreeNode*> recur(int start, int end){
	vector<TreeNode*> V;
	// if start > end then it's not a BST
	if(start > end) {
		V.push_back(nullptr);
		return V;
	}
	
	// think of the easy case where start == end, then tmp->left and tmp->right both are nullptrs and only one possible combination
	for(int i=start; i <= end; i++) {
		// subtree to the left of the node with value i
		vector<TreeNode*> left = recur(start, i-1);
		// ... to the right ....
		vector<TreeNode*> right = recur(i+1, end);

		// push_back each combination of left and right
		for(auto& l : left) {
			for(auto& r : right) {
				TreeNode* tmp = new TreeNode(i);
				tmp->left = l;
				tmp->right = r;
				V.push_back(tmp);
			}
		}
	}
	return V;
}
```
## Beautiful Triplets (Hackerrank)
* return the number of triplets where i < j < k and a[j] - a[i] == a[k] - a[j] = d
* given array in ascending order
* cool hashmap trick!
```c++
int beautifulTriplets(int d, vector<int> arr) {
	unordered_map<int, int> m;
	int cnt = 0;
	for (int a : arr) {
		m[a] += 1;
		// already in ascending order, therfore a-2*d always comes before a-d
		// don't multiply by the current num, we assume it be 1 each iteration
		cnt += m[a-d]*m[a-2*d];
	}
	return cnt;
}
```
## Sum Root to Leaf Numbers
* fun recursion
* although guaranteed that the first root can't be a nullptr, still need to check if root == nullptr in recur because maybe one leaf is nullptr
```c++
int recur(TreeNode* root, int val) {

	// take care of the case where a node has only a child
	// or else would be accessing nullptr later
	if(root == nullptr) {
		return 0;
	}
	val = 10*val + root->val;
	
	// take care of the case where a node has no child
	// or else would go on and return 0+0
	if(root->left == nullptr && root->right == nullptr) {
		return val;
	}
	return recur(root->left, val) + recur(root->right, val);
}

int sumNumbers(TreeNode* root) {
	return recur(root, 0);
}
```
## Longest Consecutive Sequence
* cool O(N) solution with hashmap
```c++
int longestConsecutive(vector<int>& nums) {
	// O(n) with map
	unordered_map<int, int> map; //key is num, value is length
	int res = 0;
	for(int num : nums){
	
		// if met a duplicate
		if(map.find(num) != map.end()) continue; 
		int left = map.find(num - 1) == map.end() ? 0 : map[num - 1];
		int right = map.find(num + 1) == map.end() ? 0 : map[num + 1];
		
		// add the # of left and right consecutive elements and itself
		int len = left + right + 1;
		
		res = max(res, len);
		map[num] = len;
		
		// set the left bound and right bound to len as well
		// so when accessing these elemens later we know they have this many consecutive elements
		if(left > 0) map[num - left] = len;
		if(right > 0) map[num + right] = len;
	}
	return res;
}
```
## Add Two Numbers
* The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.
```c++
ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
	int sum = 0;
	ListNode *ans = nullptr, *curr = nullptr;
	bool assigned = false;  

	// sum > 0 important! if both pointers null sum can be > 0 bc of carry
	while(l1 != nullptr || l2 != nullptr || sum > 0) {
		if(l1 != nullptr) {
			sum+=l1->val;
			l1 = l1->next;
		}
		if(l2 != nullptr) {
			sum+=l2->val;
			l2 = l2->next;
		}
		
		// cool trick to add numbers in reverse order
		ListNode* tmp = new ListNode(sum%10);
		sum/=10;
		if(!assigned) {
			ans = curr = tmp;
			assigned = true;
		}
		else {
			curr->next = tmp;
			curr = curr->next;
		}
	}
	return ans;
}
```
## 3Sum Closest
* Given an array nums of n integers and an integer target, find three integers in nums such that the sum is closest to target. Return the sum of the three integers.
* two pointers (lo, hi) + mid --> O(N^2) instead of O(N^3)
```c++
int threeSumClosest(vector<int>& nums, int target) {
	sort(nums.begin(), nums.end());
	int lo=0, mid=1, hi=nums.size()-1, min=INT_MAX;
	while(lo < hi) {
		// reset hi here instead of at the end of the while loop to save some unnecessary iterations
		hi=nums.size()-1;
		while(mid > lo && mid < hi) {

			if(abs(nums[lo] + nums[mid] + nums[hi]-target) < abs(min))
				min = nums[lo] + nums[mid] + nums[hi]-target;
			if(nums[lo] + nums[mid] + nums[hi] > target)
				hi--;
			// even if sum == target, increment so not stuck in loop
			else
				mid++;
		}
		lo++;
		// adjust mid accordingly
		mid = lo+1;
	}
	return target+min;
}
```
## Generate Parentheses
* recursion with leap of faith (test out with n=2)
```c++
void recur(int m, int n, string s, vector<string>& v) {

	// both '(' and ')' used up
	if(m==0 && n==0) {
		v.push_back(s);
		return;
	}
	
	// exceed the number of allowed ()
	if(m < 0 || n < 0) 
		return;
	
	// can always put a '(' first
	recur(m-1, n, s+'(', v);
	
	// ensure that it's well-formed
	if(m<n)
		recur(m, n-1, s+')', v);
}

vector<string> generateParenthesis(int n) {
	vector<string> res;
	recur(n, n, "", res);
	return res;
}
```
## Sherlock and Anagrams
* Anagrams: strings composed of the same letters
* Find the number of pairs of anagrams
* Keep track of the combination by summing the previous numbers (e.g. if 3 anagrams, # of combination is 3 bc 1+2)
```c++
int sherlockAndAnagrams(string s) {
	int size = s.size(), count=0;
	unordered_map<string, int> m;
	for(int i=0; i < size; i++) {
		for(int j=1; j <= size-i; j++) {
			string tmp = s.substr(i, j);
			sort(tmp.begin(), tmp.end());
			auto it = m.find(tmp);
			// Good way to keep track of combination!! Don't write an extra function for it
			if(it != m.end()) {
				count+=it->second;
				it->second++;
			}
			else {
				m[tmp]++;
			}
		}
	}
	return count;
}
```
## Pow(x, n)
* weird problem(?
* recursive and bit operation solutions
```c++
double myPow(double x, int n) {
	if(n == 0) return 1;
	bool isNeg = n < 0;
	double ret=1;
	n = abs(n);

	while(n > 0) {
		// if odd power, multiply by itself
		if(n & 1) {
			ret*= x;
		}
		n = n >> 1;
		// calculate the even power
		x = x*x;
	}

	return isNeg? 1/ret : ret;
}
```
```c++
double helper(double x, int n) {
	if(n == 0)       // anything raised to 0 is 1
		return 1;
	if(n == 1)      // X^1  =  X
		return x;

	double temp = helper(x, n/2);  // reach the bottom first, then traverse back up
	if(n % 2==0)
		return temp * temp;
	else
		return temp * temp * x;
}

double myPow(double x, int n) {
	if(n>0)
		return helper(x,n);
	else
		return 1/helper(x, abs(n));   // if n<0, x^n = 1/(x^n)
}
```
## Highest Value Palindrome (Hackerrank)
* lil tricky logic
* input:
    * string s: a string representation of an integer
    * int n: the length of the integer string
    * int k: the maximum number of changes allowed
```c++
string highestValuePalindrome(string s, int n, int k) {

	int lives=k;
	vector<bool> mod(n,false);  

	for(int i=0; i < n/2; i++) {
		if(s[i] != s[n-i-1]) {
			lives--;
			mod[i] = true;

			if(s[i] > s[n-i-1]) {
				s[n-i-1] = s[i];
			}
			else {
				s[i] = s[n-i-1];
			}
		}
		if(lives < 0) return "-1";
	}

	int j=0;
	while(lives > 0 && j < n/2) {
		// if s[j] == 9 but the corresponding doesn't, doesn't matter because that's lower priority
		if(s[j] != '9') {
		
			// this if statement inside the upper if, only at this point know the value is going to be changed therefore possibly increment lives
			// add 1 to lives so don't "double change" the value, increment it does the job
			if(mod[j]) {	
				lives++;
			}
			
			// have to check again bc lives can be 1 here
			if(lives >= 2) {
				s[j] = s[n-j-1] = '9';
				lives-=2;
			}
		}
		j++;
	}

	// change the middle number to 9 if has lives left
	if(n%2 == 1 && lives > 0) {
		s[n/2] = '9';
	}

	return s;
}
```
## Reverse Linked List II
* keep track of four pointers
* prev starts out at rf
```c++
ListNode* reverseBetween(ListNode* head, int left, int right) {
	if(left == right){
		return head;
	}
	
	ListNode *lft=head,*lp=NULL;
	int lc=1;
	//MOVING TOWARDS LEFT POINTER
	while(lc!=left){
		lp = lft;
		lft = lft->next;
		lc++;
	}
	
	ListNode *rgt = head;
	ListNode *rp  = rgt->next;
	int rc=1;
	//MOVING TOWARDS RIGHT POINTER
	while(rc!=right){
		rgt = rgt->next;
		rp  = rgt->next;
		rc++;
	}
	
	//LINKED LIST REVERSE LOGIC
	ListNode *curr=lft,*prev=rp,*fwd;
	while(curr!=rp){
		fwd = curr->next;
		curr->next = prev;
		prev = curr;
		curr = fwd;   
	}
	
	//IF LEFT IS OTHER THAN FIRST THEN IT WILL POINT TO RIGHT POINTER
	if(lp) lp->next = rgt;
		
	//IF LEFT IS FIRST ELEMENT THEN HEAD WILL POINT TO RIGHT POINTER
	// bc left would be null
	if(left == 1) head = rgt;
	
	return head;

}
```
![reverse_linked_list](https://user-images.githubusercontent.com/50003319/118137841-460e3000-b3cb-11eb-88a3-39eb2c1f1576.png)

## Flatten Binary Tree to Linked List
* a little abstract recursion
* use a [1, 2, 3] tree to test out
```c++
void flatten(TreeNode* root) {
	if(root == nullptr) return;

	// save the current right node, because will be replaced by left later
	TreeNode* rightNode=root->right;

	// traverse left first
	flatten(root->left);

	// traverse the right side
	flatten(rightNode);

	// move the left side to the right side
	root->right = root->left;
	root->left = nullptr;

	// traverse to the bottom of the new right side, above rightNode
	while(root->right) {
		root = root->right;
	}

	// connect the left and right
	root->right = rightNode;
}
```
## Populating Next Right Pointers in Each Node
* level-order traversal--> queue
```c++
Node* connect(Node* root) {
	if(root==nullptr) return root;

	queue<Node*> q;
	q.push(root);

	while(!q.empty()) {
		int nodeCount = q.size();
		while(nodeCount-- > 0) {
			Node* tmp = q.front();
			q.pop();

			if(nodeCount > 0) {
				tmp->next = q.front();
			} 

			if(tmp->left) q.push(tmp->left);
			if(tmp->right) q.push(tmp->right);
		}
	}
	return root;
}
```
## Reorder List
* Reorder the list to be on the following form: L0 --> Ln --> L1 --> Ln-1
* use stack
* push all nodes, easier
```c++
void reorderList(ListNode* head) {
	// rule out nullptr, one node, two nodes
	if(!head||!head->next||!head->next->next)
		return;
	stack<ListNode*> s;
	ListNode* p=head;
	int n=0;
	while(p!=NULL){
		s.push(p);
		n++;
		p=p->next;
	}
	p=head;
	for(int i=0;i<n/2;i++){
		ListNode* e=s.top();
		s.pop();
		e->next=p->next;
		p->next=e;
		p=p->next->next;
	}
	p->next=NULL;
}
```
## Minimum Depth of Binary Tree
* !A != !B is XOR boolean implementation. Can't simply do A != B, have to convert A and B to boolean values first
* The minimum depth is the number of nodes along the shortest path from the root node down to the nearest leaf node.
```c++
int minDepth(TreeNode* root) {
	if(root==nullptr) 
		return 0;
	// deal with cases where only one child is nullptr
	if(!root->left != !root->right) 
		return 1+max(minDepth(root->left), minDepth(root->right));
	else 
		return 1+min(minDepth(root->left), minDepth(root->right));
}
```
* alternative with faster runtime
```c++
int traverse(TreeNode* root) {
	// only reach here if only left or right child is nullptr
	if(root==nullptr) 
		return INT_MAX-1;
	// if leaf node
	if(root->left ==nullptr && root->right==nullptr) 
		return 1;
	else 
		return 1+min(traverse(root->left), traverse(root->right));
}

int minDepth(TreeNode* root) {
	if(root==nullptr) return 0;
	return traverse(root);
}
```
## Binary Tree Postorder Traversal with stack
```c++
vector<int> postorderTraversal(TreeNode* root) {
	vector<int> res; //a vector to return
	if(!root) return res;  //edge cases

	stack<TreeNode*> s;
	s.push(root);

	// push back the values in reverse order(top,right,left)
	while(!s.empty()) {
		TreeNode* tmp = s.top();
		s.pop();
		res.push_back(tmp->val);
		if(tmp->left) s.push(tmp->left);
		if(tmp->right) s.push(tmp->right);
	}

	reverse(res.begin(), res.end());
	return res;
}
```
## Path Sum
* Given the root of a binary tree and an integer targetSum, return true if the tree has a root-to-leaf path such that adding up all the values along the path equals targetSum
```c++
bool hasPathSum(TreeNode* root, int targetSum) {
	// reach here if at leaf node and targetSum != 0 or only one child is nullptr, which is not leaf and return false
	if(root==nullptr) 
		return false;
	targetSum-=root->val;
	if(!root->left && !root->right && targetSum==0) 
		return true;
	return hasPathSum(root->left, targetSum) || hasPathSum(root->right, targetSum);
}
```
## Path Sum II
* Given the root of a binary tree and an integer targetSum, return all root-to-leaf paths where each path's sum equals targetSum.
* Note the pop_back()!
```c++
vector<vector<int>> v;
void tra(TreeNode* root, int targetSum, vector<int>& tmp) {
	
	// take care of the cases at leaf nodes and when one child is nullptr
	if(root==nullptr) return;

	targetSum-= root->val;
	tmp.push_back(root->val);
	if(!root->left && !root->right && targetSum==0) {
		v.push_back(tmp);
	}
	
	tra(root->left, targetSum, tmp);
	tra(root->right, targetSum, tmp);
	// pop_back is important, get rid of used values
	tmp.pop_back();
}

vector<vector<int>> pathSum(TreeNode* root, int targetSum) {
	vector<int> vec;
	tra(root, targetSum, vec);
	return v;
}
```
## Convert Sorted List to Binary Search Tree
1. use fast and slow pointers to find the middle element!!
	* brilliantuse of the extra parameter tail=nullptr
2. use vector to store the values first
```c++
TreeNode* sortedListToBST(ListNode* head, ListNode* tail=nullptr) {
	if(head == tail) return nullptr;
	ListNode* slow=head, *fast = head;
	
	while(fast != tail && fast->next != tail) {
		slow = slow->next;
		fast = fast->next->next;
	}
	
	TreeNode* root = new TreeNode(slow->val);
	// early return to save some runtime
	if(slow == fast) return root;
	root->left = sortedListToBST(head, slow);
	root->right = sortedListToBST(slow->next, tail);
	return root;
}
```
```c++
TreeNode* traverse(vector<int>& v, int start, int end) {
	if(start > end) return nullptr;
	// early return to save some runtime
	if(start == end) {
		return new TreeNode(v[start]);
	}
	
	int mid = (start+end) / 2;
	TreeNode* root = new TreeNode(v[mid]);
	root->left = traverse(v, start, mid-1);
	root->right = traverse(v, mid+1, end);
	return root;
}

TreeNode* sortedListToBST(ListNode* head) {
	vector<int> v;
	while(head != nullptr) {
		v.push_back(head->val);
		head = head->next;
	}
	return traverse(v, 0, v.size()-1);
}
```
## Bitwise AND of Numbers Range
* THINK!
* find the first different bit reading from the MSB of left and right
	* the following bits starting at the different bit would all be 0's when ANDed together
* return the same bits reading the MSB
* note: if left and right have different number of bits, result is 0
```c++
int rangeBitwiseAnd(int left, int right) {
	int count=0;
	while(left!=right){
		left=left>>1;
		right=right>>1;
		count++;
	}
	return left<<count;
}
```
## Remove Duplicates from Sorted List II
* Given the head of a sorted linked list, delete all nodes that have duplicate numbers, leaving only distinct numbers from the original list. Return the linked list sorted as well
* Don't change tmp until we're sure there's no duplicate
```c++
ListNode* deleteDuplicates(ListNode* head) {
	ListNode* dummy = new ListNode();
	dummy->next = head;
	ListNode* tmp = dummy;
	while(head){
		while(head->next && head->val == head->next->val) {
			head=head->next;
		}
		
		// smart if-else statement
		// only change tmp when know for sure there's no duplicates followed, i.e. head didn't get incremented
		// else simply redirect tmp to head->next
		if(tmp->next == head)
			tmp = tmp->next;
		else
			tmp->next = head->next;
		head = head->next;
	}
	return dummy->next;
}
```
## Recover Binary Search Tree
* prev represents previous node in inorder traversal
```c++
// some global variables
TreeNode *prev=NULL,*first=NULL,*second=NULL;

void solve(TreeNode *root) {
	if(root==NULL)
		return;
		
	solve(root->left);
	if(prev != NULL && root->val < prev->val) {
		if(first==NULL)
			first=prev;
		second=root;
	}
	
	prev=root;
	solve(root->right);
}

void recoverTree(TreeNode* root) {
	solve(root);
	swap(first->val,second->val);
}
```
## Edit Distance
* Given two strings word1 and word2, return the minimum number of operations required to convert word1 to word2.
* You have the following three operations permitted on a word:
    * Insert a character
    * Delete a character
    * Replace a character
* DP!!! Make a 2D array where each element represents the smallest number of change from the previous state.
* if the current character is the same, take the number from the upper left corner, else take the minimum of top, elft upper left corner and plus 1
```c++
int shortestSteps(string& s1, string& s2) {
	vector<vector<int>> v(s1.size()+1, vector<int>(s2.size()+1, 0));
	for(int i=0; i < s2.size()+1; i++) {
		v[0][i] = i;
	}
	for(int i=0; i < s1.size()+1; i++) {
		v[i][0] = i;
	}
	for(int i=1; i < s1.size()+1; i++) {
		for(int j=1; j < s2.size()+1; j++) {
			if(s1[i-1] == s2[j-1]) v[i][j]=v[i-1][j-1];
			else {
				int lowest=min(min(v[i-1][j], v[i][j-1]), v[i-1][j-1]);
				v[i][j] = lowest+1;
			}
		}
	}

	return v[s1.size()][s2.size()];
}
```
## Longest Increasing Path
* Given an m x n integer matrix, return the length of the longest increasing path in matrix. From each cell, you can either move left, right, up, or down. you may not move diagonally or outside the boundary.
* DFS + graph, tricky!!!
```c++
void dfs(int i, int j, vector<vector<int>>& matrix, vector<vector<int>>& memo) {
	vector<vector<int>> dirs = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};

	memo[i][j] = 0;
	// check the four direction of the current element
	for(int k=0; k < 4; k++) {
		int r = i + dirs[k][0];
		int c = j + dirs[k][1];

		// if out of bound
		if(r < 0 || r >= matrix.size() || c < 0 || c >= matrix[0].size()) continue;

		// increasing
		if(matrix[i][j] > matrix[r][c]) {

			// if memo not filled at r, c
			if(memo[r][c] == -1) {
				dfs(r, c, matrix, memo);
			}

			// the 1 accounts for counting the current element
			memo[i][j] = max(memo[i][j], 1 + memo[r][c]);
		}
	}
}

int longestIncreasingPath(vector<vector<int>>& matrix) {
	vector<vector<int>> memo(200, vector<int>(200, -1));

	int path=0;
	for(int i=0; i < matrix.size(); i++) {
		for(int j=0; j < matrix[i].size(); j++) {
			// if hasn't been set yet
			if(memo[i][j] == -1) {
				dfs(i, j, matrix, memo);
			}
			path = max(path, memo[i][j]);
		}
	}
	//	for(int i=0; i < matrix.size(); i++) {
	//		for(int j=0; j < matrix[i].size(); j++) {
	//			cout << memo[i][j] << " ";
	//		}
	//		cout << endl;
	//	}

	// path counts the edges, plus 1 to account for the number of elements
	return path+1;
}
```
## Cheapest Price
* There are n cities connected by some number of flights. You're given an arrya flights where flights[i] = [fromi, toi, pricei] indicates that there is a flight from city fromi to city toi with cost pricei. You're also given three integers src, dst, and k, return the cheapest price from src to dst with at most k stops. If there is no such route, return -1.
![cheapestPrice](https://user-images.githubusercontent.com/50003319/119598960-a036ca00-bda9-11eb-84a4-04a31560c6b0.png)
* in the pic, e-1 refers to the previous column with one less edge
```c++
int cheapestPrice(int n, vector<vector<int>>& flights, int src, int dst, int k) {
	int MAX_PRICE = 1e6;

	vector<vector<pair<int, int>>> graph(n+1);
	for(auto flight : flights) {
		pair<int, int> edge = make_pair(flight[1], flight[2]);
		graph[flight[0]].push_back(edge);
	}

	// dp[i][j] represents the shortest path to node i using max of j edges
	vector<vector<int>> dp(n+1, vector<int>(n+1, MAX_PRICE));

	// if src == dst the distance is zero no matter how many edges are available.
	for(int i=0; i <= n; i++) {
		dp[src][i] = 0;
	}

	// have to know ki-1 before building ki, have to know values wtih fewer number of stops first
	for(int ki = 1; ki <= n; ki++){
		for(int srcj = 0; srcj < n; srcj++) {
			// traverse all the neighbors
			for(auto edge : graph[srcj]) {
				int dest = edge.first;
				int price = edge.second;

				// minimum of its previous node and the (shortest path to its neighbor + the weight of the node)
				dp[dest][ki] = min(dp[dest][ki], min(dp[dest][ki-1], dp[srcj][ki-1] + price));
			}
		}
	}
	//	for(int i=0; i < n+1; i++) {
	//		for(int j=0; j < n+1; j++) {
	//			cout << dp[i][j] << " ";
	//		}
	//		cout << endl;
	//	}

	return (dp[dst][k+1] >= MAX_PRICE) ? -1 : dp[dst][k+1];
}
```
## All Paths From Source to Target
* Given a directed acyclic graph (DAG) of n nodes labeled from 0 to n - 1, find all possible paths from node 0 to node n - 1, and return them in any order.
* The graph is given as follows: graph[i] is a list of all nodes you can visit from node i (i.e., there is a directed edge from node i to node graph[i][j])
```c++
vector<vector<int>> ret;
void traverse(vector<vector<int>>& graph, vector<int>& tmp, int n, int pos) {
	tmp.push_back(pos);
	if(pos==n) ret.push_back(tmp);
	for(int i=0; i < graph[pos].size(); i++) {
		traverse(graph, tmp, n, graph[pos][i]);
	}
	tmp.pop_back();
}

vector<vector<int>> allPathsSourceTarget(vector<vector<int>>& graph) {
	int n=graph.size()-1;
	vector<int> v;
	traverse(graph, v, n, 0);
	return ret;
}
```
## Count Servers that Communicate
* You are given a map of a server center, represented as a m * n integer matrix grid, where 1 means that on that cell there is a server and 0 means that it is no server. Two servers are said to communicate if they are on the same row or on the same column.
```c++
unordered_map<int, vector<int>> mpr, mpc;
int ans=0;

int countServers(vector<vector<int>>& grid) {    
	int m=grid.size(), n=grid[0].size();

	vector<vector<bool>> visited(m, vector<bool>(n, false));
	for(int i=0; i < m; i++) {
		for(int j=0; j < n; j++) {
			if(grid[i][j]) {
			mpr[i].push_back(j);
			mpc[j].push_back(i);
			}
		}
	}

	for(auto& i : mpr) {
		for(auto& j : i.second) {
			// account for cases where ans is odd, otherwise taken care by dfs
			// typically the first traversal if any, because none has been visisted yet
			if(!visited[i.first][j] && dfs(i.first, j, visited)) {
				ans++;
			}
		}
	}
	return ans;
}

bool dfs(int r, int c, vector<vector<bool>>& visited) {
	visited[r][c]=true;
	bool isPos=false;

	// traverse the row map
	for(auto& i : mpr[r]) {
		// check visisted so don't double count
		if(!visited[r][i]) {
			ans++;
			// recursion until all visited
			dfs(r, i, visited);
			isPos=true;
		}          
	}

	// traverse the column map
	for(auto& i : mpc[c]) {
		if(!visited[i][c]) {
			ans++;
			dfs(i, c, visited);
			isPos=true;
		}
	}
	return isPos;
}
```
## Lowest Common Ancestor (Hackerrank)
* surprisingly easy!? Cool concept
* find the first encounter where root->val < v1 && root->val > v2
```c++
Node *lca(Node *root, int v1, int v2) {
	while (root != nullptr) {
		// know both nodes have values less than root, there must be a common ancestor to the left of the root
		if (root->data > v1 && root->data > v2) {
			root = root->left;
		} 
		else if (root->data < v1 && root->data < v2) {
			root = root->right;
		} 
		// first encounter that satisfies root->data > v1 && root->data < v2 is the answer
		else {
			break;
		} 
	}
	return root;
}
```
## Jim and the Skyscrapers (Hackerrank)
* brute force (nested for loop) won't work
* elegent approach: stack with pair!?
```c++
typedef unsigned long long ull;
	ull solve(vector<int> arr) {
	ull count=0;
	int n = arr.size();
	stack<pair<ull, ull>> s;
	for(int i=0;i<n;i++) {
		// remove unnecessary elements
		while(!s.empty() && arr[i] > s.top().first)
			s.pop();
		if(!s.empty() && arr[i]==s.top().first)
			// crucial!!! s.top().second++ accounts for > 2 same entries (e.g. three 3's should be counted 3 times)
			count+=s.top().second++;
		else
			s.push(make_pair(arr[i],1));
	}
	return count*2;
}
```
## Is This a Binary Search Tree? (Hackerrank)
* duplicates not allowed!
```c++
// inorder traversal approach (inefficient?)
void inorder(Node* root, vector<int>& v) {
	if(root == nullptr) return;
	inorder(root->left, v);
	v.push_back(root->data);
	inorder(root->right, v);
}

bool checkBST(Node* root) {
	vector<int> ret;
	inorder(root, ret);
	int s = ret.size();
	
	for(int i=0; i < s-1; i++) {
		if(ret[i] >= ret[i+1]) 
			return false;
	}
	return true;
}
```
```c++
// smart recursion
// extra param min, max help keep track of the should-be upper/lower bound, if violated then false
bool check(Node* root, int min, int max) {
	if (root != nullptr) {
		if (root->data >= max || root->data <= min) {
			return false;
		}
		else {
			return check(root->left, min, root->data) & check(root->right, root->data, max);
		}   
	}
	else {
		return true;
	}
}

bool checkBST(Node* root) {
	return check(root, 0, INT_MAX);
}
```
## Find Maximum Index Product (Hackerrank)
* find the greatest product of indices i,j 
	* where i is the left closest index that satisfies a[i] > a[curr]
	* where j is the right closest index that satisfies a[j] > a[curr]
* store the desired index to the corresponding index in left/right
* one of the terms in right[i]*left[i] for the left/right most element is 0, so always yields 0
* stack used to reduce the elements stored, works because ask for *closest* index
```c++
long long solve(vector<int> arr) {
	stack<int> Stack;
	int n = arr.size();
	vector<int> left(n, 0) , right(n, 0);
	
	for(int i=0 ; i<n; i++) {
		while(!Stack.empty() and arr[Stack.top()] <= arr[i])
			Stack.pop();
		if(!Stack.empty())
		// access the stack top, so always the closest on the left
			left[i] = Stack.top() + 1;
		Stack.push(i);
	}

	// clear the stack for right vector
	while(!Stack.empty()) Stack.pop();
	
	// have to go from right to left to keep track of closest on the right
	for(int i=n-1 ; i>=0 ; i--) {
		while(!Stack.empty() and arr[Stack.top()] <= arr[i])
			Stack.pop();
		if(!Stack.empty())
			right[i] = Stack.top() + 1;
		Stack.push(i);
	}

	long long ans = 0;

	for(int i=0 ; i<n ; i++) {
		ans = max(ans ,(long long)left[i] * right[i]);
	}
	
	return ans;
}
```
## Maximum Element
* rule:
	* 1: Push the element x into the stack.
	* 2: Delete the element present at the top of the stack.
	* 3: Print the maximum element in the stack.
```c++
vector<int> getMax(vector<string> operations) {
	vector<int> ret;
	stack<int> s;
	int size = operations.size();
	string op, num;

	for(int i=0; i < size; i++) {
		istringstream ss(operations[i]);
		ss >> op;
		ss >> num;
		if(op == "1") {
			int p = stoi(num);
			if(s.empty()) s.push(p);
			// crucial!!! push the current max element of the array each iteration, possibly pushing repeating numbers
			else s.push(max(p, s.top()));
		}
		if(op == "2") {
			if(!s.empty()) s.pop(); 
		}
		if(op == "3") {
			ret.push_back(s.top());
		}
	}
	return ret;
}
```
## Find the Town Judge
* graph! But use two vectors to store indegrees and outdegrees
* In a town, there are n people labelled from 1 to n.  There is a rumor that one of these people is secretly the town judge.
* If the town judge exists, then:
	1. The town judge trusts nobody.
	2. Everybody (except for the town judge) trusts the town judge.
	3. There is exactly one person that satisfies properties 1 and 2.
```c++
int findJudge(int n, vector<vector<int>>& trust) {
	vector<int> truster(n+1, 0);
	vector<int> trusted(n+1, 0);

	for(int i=0; i < trust.size(); i++) {
		truster[trust[i][0]]++;
		trusted[trust[i][1]]++;
	}

	for(int i=1; i < n+1; i++) {
		if(truster[i]==0 && trusted[i] == n-1) return i;
	}

	return -1;
}
```
## 
```c++
// BFS
// TC = 108ms

class Solution {
public:
	int networkDelayTime(vector<vector<int>>& times, int n, int k) {
		vector<pair<int,int>> adj[n+1];
		for(int i=0;i<times.size();i++)
			adj[times[i][0]].push_back({times[i][1],times[i][2]});
			
		vector<int> dist(n+1,INT_MAX);
		queue<int> q;
		q.push(k);
		// distance to itself is 0
		dist[k]=0;
		
		while(!q.empty()) {
			int t=q.front();
			q.pop();
			for(pair<int,int> it:adj[t]) {
				// if source + weight < current distance
				if(dist[it.first]>dist[t]+it.second) {
					dist[it.first]=dist[t]+it.second;
					q.push(it.first);
				}
			}
		}
		
		int res=0;
		for(int i=1;i<=n;i++){
			if(dist[i]==INT_MAX)
				return -1;
			res=max(res,dist[i]);
		}
		return res;
	}
};
```
```c++
// Dijkstra Algorithm
// TC = 112ms

class Solution {
public:
	int networkDelayTime(vector<vector<int>>& times, int n, int k) {
		vector<vector<pair<int,int>>> adj(n+1);
		for(int i=0;i<times.size();i++)
			adj[times[i][0]].push_back({times[i][1],times[i][2]});
		vector<int> dist(n+1,INT_MAX);
		
		priority_queue<pair<int,int>,vector<pair<int,int>>,greater<pair<int,int>>> pq;
		pq.push({0,k});
		dist[k]=0;
		while(!pq.empty()) {
			pair<int,int> t = pq.top();
			pq.pop();
			
			for(pair<int,int> it : adj[t.second]) {
				if(dist[it.first] > t.first+it.second) {
					dist[it.first]=t.first+it.second;
					// weight is the first element of pair
					pq.push({dist[it.first],it.first});
				}
			}
		}
		
		int res=0;
		for(int i=1;i<=n;i++) {
			if(dist[i]==INT_MAX)
				return -1;
			res=max(res,dist[i]);
		}
		return res;
	}
};
```
```c++
// Bellman-Ford Algorithm
// TC = 208ms

class Solution {
public:
	int networkDelayTime(vector<vector<int>>& times, int n, int k) {
		vector<int> dist(n+1,INT_MAX);
		dist[k]=0;

		for(int i=0;i<n;i++) {
			bool flag=false;
			for(auto node:times) {
				int src = node[0];
				int des = node[1];
				int time = node[2];
				if(dist[src] != INT_MAX && dist[des] > dist[src]+time) {
					dist[des]=dist[src]+time;
					flag=true;
				}
			}
			// if nothing has been changed then done making the dist array
			if(flag==false)
				break;
		}
		
		int res=0;
		for(int i=1;i<=n;i++) {
			if(dist[i]==INT_MAX)
				return -1;
			res=max(res,dist[i]);
		}
		return res;
	}
};
```
## Keys and Rooms
* both BFS and DFS work!
```c++
// DFS
void dfs(vector<vector<int>>& rooms, vector<bool>& visited, int v, int &count) {

	visited[v] = true;
	count++;

	for(auto i : rooms[v]) {
		if(!visited[i])
		dfs(rooms, visited, i, count);
	}
}    
bool canVisitAllRooms(vector<vector<int>>& rooms) {

	int n = rooms.size();
	vector<bool> visited(n, false);
	int count = 0;

	dfs(rooms, visited, 0, count);

	return count == n;
}
```
```c++
// BFS
bool canVisitAllRooms(vector<vector<int>>& rooms) {
	int n = rooms.size();
	vector<bool> canEnter(n, false);
	canEnter[0] = true;

	queue<int> q;
	q.push(0);

	while(!q.empty()) {
		int num = q.front();
		q.pop();
		for(auto& i : rooms[num]) {
			if(!canEnter[i]) {
				canEnter[i] = true;
				q.push(i);
			}
		}
	}

	for(int i=0; i < n; i++) {
		if(canEnter[i] == false) return false;
	}
	return true;
}
```
##  Minimum Number of Vertices to Reach All Nodes
* Find the smallest set of vertices from which all nodes in the graph are reachable. 
* if there's zero indegrees, then it must be included in the set of vertices
```c++
vector<int> findSmallestSetOfVertices(int n, vector<vector<int>>& edges) {
	
	vector<int> v(n, 1);
	for(int i=0; i < edges.size(); i++) {
		v[edges[i][1]] = 0;
	}

	vector<int> ret;
	for(int i=0; i < n; i++) {
		if(v[i] == 1) ret.push_back(i);
	}

	return ret;
}
```
## Redundant Connection
* In this problem, a tree is an undirected graph that is connected and has no cycles.
* Return an edge that can be removed so that the resulting graph is a tree of n nodes. If there are multiple answers, return the answer that occurs last in the input.
```c++
vector<int>parent;
vector<int> findRedundantConnection(vector<vector<int>>& edges) {
	// can't declare a vector with size in class! use assign here
	parent.assign(1001,-1);
	vector<int>ans;
	for(int i=0;i<edges.size();i++) {
		int x=find(edges[i][0]);
		int y=find(edges[i][1]);
		// if x == y then there's a cycle
		if(x==y) {
			// clear answer to get the last desired adge
			ans.clear();
			ans.push_back(edges[i][0]);
			ans.push_back(edges[i][1]);
		}
		else
			ArrUnion(x,y);
	}
	return ans;
}

// find the root?? If have the same root then in the same set
int find(int val) {
	while(parent[val]!=-1)
		val=parent[val];
	return val;
}

// link two elements
void ArrUnion(int x,int y) {
	parent[x]=y;
}
```
## Find Center of Star Graph
* There is an undirected star graph consisting of n nodes labeled from 1 to n. A star graph is a graph where there is one center node and exactly n - 1 edges that connect the center node with every other node.
* trick: find the same number in the first two elements
```c++
int findCenter(vector<vector<int>>& edges) {
	if(edges[0][0]==edges[1][0]||edges[0][0]==edges[1][1])
		return edges[0][0];

	else
		return edges[0][1];
}
```
## Binary Subarrays With Sum
* Given a binary array nums (only contains 0s and 1s) and an integer goal, return the number of non-empty subarrays with a sum goal.
* tricky logics??
```c++
int numSubarraysWithSum(vector<int>& nums, int goal) {
	int start=0,end=0,pre=0,sum=0,ans=0;

	while(end < nums.size()) {
		if(nums[end]==1) {
			sum++;
		}

		if(sum > goal) {
			// only skip one 1
			while(nums[start]!=1) {
				start++;
			}
			start++;
			// the run is broken, have to count pre again
			pre=0;
			sum--;
		}

		// if there's leading zeroes, it's ok to include
		while(start < end && nums[start] == 0) {
			pre++;
			start++;
		} 

		// trailing zeroes accounted here, second condition necessary
		if(sum==goal && start<=end) {
			ans+=pre+1;
		}
		end++;
	}
	
	return ans;
}
```
## Queue using Two Stacks
* lil tricky in case 2 and 3
* rules:
	* 1 x: Enqueue element  into the end of the queue.
	* 2: Dequeue the element at the front of the queue.
	* 3: Print the element at the front of the queue.
```c++
stack<int> reverse, ordered;
int n;
cin >> n;
while(n--) {
	int op;
	cin >> op;
	if(op == 1) {
		int num;
		cin >> num;
		reverse.push(num);
	}
	else {
		// if nothing in ordered to be outputted, move stuff from reverse
		if(ordered.empty()) {
			while(!reverse.empty()) {
				ordered.push(reverse.top());
				reverse.pop();
			}
		}
		
		// have stuff to be popped, not an else statement cuz both if's can be executed
		if(!ordered.empty()) {
			if(op == 2) ordered.pop();
			if(op == 3) cout << ordered.top() << endl;
		}
	}
}
```
## 4Sum II
* O(N^2) solution with hashmap!!
* Given four integer arrays nums1, nums2, nums3, and nums4 all of length n, return the number of tuples (i, j, k, l) such that:
	* 0 <= i, j, k, l < n
	* nums1[i] + nums2[j] + nums3[k] + nums4[l] == 0
```c++
// approach 1: only one map, less memory complexity
int fourSumCount(vector<int>& nums1, vector<int>& nums2, vector<int>& nums3, vector<int>& nums4) {
	unordered_map<int, int> um;
	int ret=0;
	for(int i=0; i < nums1.size(); i++) {
		for(int j=0; j < nums2.size(); j++) {
			um[nums1[i]+nums2[j]]++;
		}
	}

	for(int i=0; i < nums3.size(); i++) {
		for(int j=0; j < nums4.size(); j++) {
			// note the '-' sign, want them to add up to 0
			if(um.find(-(nums3[i]+nums4[j]))!= um.end()) {
				ret+=um[-(nums3[i]+nums4[j])];
			}
		}
	}
	return ret;
}
```
```c++
// approach 2: two maps, one extra 2D vector, less straightforward, also O(N^3)
int fourSumCount(vector<int>& nums1, vector<int>& nums2, vector<int>& nums3, vector<int>& nums4) {
	vector<vector<int>> nums = { nums1, nums2, nums3, nums4 };
	unordered_map<int, int> cnt[2];
	
	for (int i = 0; i < 2; i++) 
		for (auto& x : nums[i*2]) 
			for (auto& y : nums[i*2 + 1]) 
				cnt[i][x + y]++;

	int res = 0;
	for (auto& j : cnt[0]) 
		// if -j.first DNE in cnt[1], res+=0
		res += cnt[1][-j.first] * j.second;

	return res;
}
```
## 2 Keys Keyboard
* There is only one character 'A' on the screen of a notepad. You can perform two operations on this notepad for each step:
	* Copy All: You can copy all the characters present on the screen (a partial copy is not allowed).
	* Paste: You can paste the characters which are copied last time.
* Given an integer n, return the minimum number of operations to get the character 'A' exactly n times on the screen.
```c++
// my solution: beats 100%!!!
int minSteps(int n) {
	if(n == 1) return 0;
	int inc=0, cur=1, ans=0;
	while(cur < n) {
		if(n % cur == 0) {
			ans+=2;
			inc=cur;
			cur+=inc;
		}
		else {
			ans++; 
			cur+=inc;
		}
	}
	return ans;
}
```
```c++
// DP solution
int minSteps(int n) {
	vector<int>dp(n+1);
	dp[0]=0;
	dp[1]=0;
	for(int i=2; i<=n; i++){
		int j=i/2;
		// find the biggest factor
		while(i % j != 0) {
			j--;
		}
		// account for how many times to paste plus one copy
		dp[i] = dp[j] + i/j;
	}
	return dp[n];
}
```
## Arithmetic Slices
* An integer array is called arithmetic if it consists of at least three elements and if the difference between any two consecutive elements is the same.
* Given an integer array nums, return the number of arithmetic subarrays of nums.
```c++
// don’t need an array, simply keep track of the longest sequence, k,  them sum up from 1…k
int numberOfArithmeticSlices(vector<int>& nums) {
	int k=0, sum=0;
	for(int i=2; i < nums.size(); i++) {
	if(nums[i-2] - nums[i-1] == nums[i-1] - nums[i])
		k++;
		else {
			sum+=k*(k+1)/2;
			// zeros k because the sequence breaks
			k=0;
		}
	}
	return sum+=k*(k+1)/2;
}
```
```c++
// DP approach, use an array
// each integer element represents the number of arithmetic sequences that ends on that index
int numberOfArithmeticSlices(vector<int>& nums) {
	int s = nums.size();
	vector<int> dp(s, 0);
	int ret=0;
	for(int i=2; i < s; i++) {
		if(nums[i-2]-nums[i-1] == nums[i-1]-nums[i]) {
			dp[i] = dp[i-1] + 1;
			ret+=dp[i];
		}
	}
	return ret;
}
```
## Search a 2D Matrix
* two binary searches, one over the last column, one over the selected row
```c++
bool searchMatrix(vector<vector<int>>& A, int target) {
	int start=0,end=A.size()-1;
	int n=A[0].size()-1,mid,prev;

	// early return if less/greater than smallest/biggest element
	if(A[0][0]>target || A[A.size()-1][A[0].size()-1] < target)return 0;
	while(start <= end) {

		mid=(start+end)/2;

		if(A[mid][n] < target) start=mid+1;
		else if(A[mid][n] > target){
			prev=mid-1;
			// if at the first row or the last term of the previous row is less than target
			if(prev==-1 || A[prev][n] < target) break;
			end=mid-1;
		}
		else return 1;
	}

	int low=0,high=A[0].size()-1;
	int avg;
	while(low <= high){
		avg=(low+high)/2;
		if(A[mid][avg]>target) high=avg-1;
		else if(A[mid][avg]<target) low=avg+1;
		else return 1;
	}
	return 0;
}
```
## Insertion Sort List
* Given the head of a singly linked list, sort the list using insertion sort, and return the sorted list's head.
* Dummy node accounts for the swapping with head cases
```c++
ListNode *insertionSortList(ListNode *head) {

	// using the dummy head method
	ListNode *newHead = new ListNode();
	newHead->next = head;

	// the pointers we need
	ListNode *prev = newHead->next;
	ListNode *traverse = prev->next;
	ListNode *frontPointer = newHead;
	ListNode *frontPointerPrev = newHead;

	// traversing the LL till we reach the end
	while (traverse != NULL) {
	int val = traverse->val;
		// if the value of current Node is smaller than the previos Node
		// we find the correct position of the current Node starting from the 'newHead->next' pointer
		// after finding the the position where we need to place the current Node we move Nodes
		if (prev->val > val) {
			frontPointerPrev = newHead;
			frontPointer = newHead->next;
			while (frontPointer->val < val) {
				frontPointerPrev = frontPointer;
				frontPointer = frontPointer->next;
			}

			// moving the pointers to place the current Node in correct Position
			prev->next = traverse->next;
			traverse->next = frontPointer;
			frontPointerPrev->next = traverse;

			// moving traverse, original traverse->next
			traverse = prev->next;
		}
		else {
			prev = traverse;
			traverse = traverse->next;
		}
	}

	return newHead->next;
}
```
## Tree : Top View
* map.insert(n): insert n if n's key is unique, doesn't overwrite the previous element with the same key
* Given a pointer to the root of a binary tree, print the top view (from left to right) of the binary tree.
* integer in the pair represents vertical distance from the root (negative values to the left of the root, positive values to the right of the root)
* **a pair and a map element (key + value) are interchangeable, first value of pair treated as key**
```c++
void topView(Node * root) {
	queue<pair<int, Node*>> q; 
	q.push({0,root});
	map<int, Node*> ans;
	// interesting syntax, same as while loop
	for(auto i=q.front(); !q.empty(); q.pop(),i=q.front()){
		if(!i.second) continue;
		ans.insert(i);
		q.push({i.first+1, i.second->right});
		q.push({i.first-1, i.second->left});
	}
	for(auto i:ans) cout << i.second->data << " ";
}
```
## 4Sum
* Given an array nums of n integers, return an array of all the unique quadruplets [nums[a], nums[b], nums[c], nums[d]] such that:
	* 0 <= a, b, c, d < n
	* a, b, c, and d are distinct.
	* nums[a] + nums[b] + nums[c] + nums[d] == target
```c++
vector<vector<int>> res;
vector<vector<int>> fourSum(vector<int>& nums, int target) {

	sort(nums.begin(),nums.end());
	int n = nums.size();
	// n-3 to make space for a, b, c
	for(int i=0; i < n-3; i++){
		// if the smallest positive already bigger than target, no chance
		// if the smallest negative bigger than target, still chance, can go lower
		if(nums[i] > 0 && nums[i] > target) break;
		
		int F=nums[i];
		
		// skip over duplicates
		if(i > 0 && F==nums[i-1]) continue;
		// n-2 to make space for b, c
		for(int a=i+1; a < n-2; a++) {
			int A = nums[a];
			
			// skip over duplicates
			// first condition to ensure checking in a's range, (e.g. if nums[i] == nums[a] == 1 doesn't count as duplicates, should avoid)
			if(a > i+1 && A == nums[a-1]) continue;
			
			int b=a+1, c=n-1;
			int target2 = target-nums[a]-nums[i];
			while(b < c) {
				int B = nums[b];
				int C = nums[c];
				int sum = B+C;
				
				if(sum==target2) {
					res.push_back({nums[i],nums[a],nums[b],nums[c]});
					// skip over duplicates
					// b < c to ensure checking for duplicate in b's/c's range
					while(b < c && nums[b]==B) b++;
					while(b < c && nums[c]==C) c--;
				} 
				else {
					if(sum > target2) c--;
					else b++;
				}
			}
		}
	}
	return res;
}
```
## 3Sum
* Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0
```c++
vector<vector<int>> threeSum(vector<int>& nums) {
	sort(nums.begin(), nums.end());
	vector<vector<int>> ret;
	int n = nums.size();

	for(int lo=0; lo < n; lo++) {
		// skip the duplicates for lo
		if(lo > 0 && nums[lo] == nums[lo-1]) continue;

		// reintialize mid and hi
		int mid = lo+1, hi = n-1;
		while(mid < hi) {
			if(nums[lo] + nums[mid] + nums[hi] == 0) {
				ret.push_back({nums[lo], nums[mid], nums[hi]});
				
				int val = nums[mid];
				// only skip duplicates when the sum is 0
				// necessarily go through while loop once since start at itself!
				while(mid < n && val == nums[mid]) mid++;
				val = nums[hi];
				while(hi >= 0 && val == nums[hi]) hi--;
			}
			else if(nums[lo] + nums[mid] + nums[hi] < 0) {
				mid++;
			}
			else {
				hi--;
			}
		}
	}
	return ret;
}
```

## Reverse a doubly linked list (Hackerrank)
* head->prev is nullptr!!!
```c++
DoublyLinkedListNode* reverse(DoublyLinkedListNode* llist) {
	DoublyLinkedListNode* temp = llist;
	DoublyLinkedListNode* newHead = llist;
	while (temp != nullptr) {
		DoublyLinkedListNode* prev = temp->prev;
		temp->prev = temp->next;
		temp->next = prev;
		newHead = temp;
		temp = temp->prev;
	}
	return newHead;
}
```
## Construct Binary Tree from Inorder and Postorder Traversal
* Use a map to track inorder positions!
```c++
map<int, int> dict;
TreeNode* buildTreeUtil(vector<int> &inorder, vector<int> &postorder, int &index, int low, int high){
	// not bounded by anything
	if(low > high){
		return NULL;
	}

	// the end of postorder is the root
	TreeNode* root = new TreeNode(postorder[index]);
	// get the inorder position of the root value in dict
	int idx = dict[postorder[index--]];
	// index reduced by one to get to the right node of root in postorder
	// every node to the right of root in dict (inorder) is bounded by idx+1 and high
	root->right = buildTreeUtil(inorder, postorder, index, idx + 1, high);
	// index always passed by reference, changes value here
	// index reduced by one again to get to the left node of root in postorder
	root->left = buildTreeUtil(inorder, postorder, index, low, idx - 1);
	return root;
}

TreeNode* buildTree(vector<int>& inorder, vector<int>& postorder) {
	int n = postorder.size();
	// put inorder vector in order
	for(int i = 0; i < n; i++){
		dict[inorder[i]] = i;
	}

	// index = end of the postorder, root is at the end of postorder traversal
	int index = n - 1;
	return buildTreeUtil(inorder, postorder, index, 0, n - 1);
}
```
## Array Manipulation (Hackerrank)
* Starting with a 1-indexed array of zeros and a list of operations, for each operation add a value to each the array element between two given indices, inclusive. Once all operations have been performed, return the maximum value in the array.
* don't actually add the numbers! keep track of where adding begins and ends
```c++
long arrayManipulation(int n, vector<vector<int>> queries) {
	vector<long> v(n+1);
	int size = queries.size();
	for(int i=0; i < size; i++) {
		long a = queries[i][0], b = queries[i][1], k = queries[i][2];
		// the adding the element
		v[a]+=k;
		// crucial to do -=k!!!!
		// stop adding the element, subtract it back to its original value
		if(b != n) v[b+1]-=k;
	}
	long max=0, x=0;
	for(int i=1; i <= n; i++) {
		// summing gets the current max cumulative sum
		x+=v[i];
		if(max < x) max = x;
	}
	return max;
}
```
## Remove Nth Node From End of List
* Given the head of a linked list, remove the nth node from the end of the list and return its head.
* Attach a dummy node to the front and nullptr to the end to account for edge cases!
```c++
ListNode* removeNthFromEnd(ListNode* head, int n) {
	vector<ListNode*> v;
	ListNode* dummy = new ListNode(0);
	dummy->next = head;
	ListNode* ptr = dummy;;
	while(ptr != nullptr) {
		v.push_back(ptr);
		ptr = ptr->next;
	}
	v.push_back(nullptr);
	int size = v.size();
	v[size-n-2]->next = v[size-n];
	return dummy->next;
}
```
```
// Two pointers approach
ListNode* removeNthFromEnd(ListNode* head, int n) {
	ListNode* dummy=new ListNode(0);
	dummy->next=head;
	ListNode* slow=dummy;
	ListNode* fast=dummy;
	// fast gets a head start by n
	for(int i=1;i<=n+1;i++) fast=fast->next;
	// want slow and fast spaced by n when fast hits nullptr --> to get the desired position n from the end of the node
	while(fast!=NULL){
		fast=fast->next;
		slow=slow->next;
	}
	slow->next=slow->next->next;
	return dummy->next;
}
```
## Maximizing XOR
* Given two integers m and n find the maximal value of a xor b written where and satisfy the following condition: m <= a <= b <= n
* Trick: find the greatest differing bit, between m and n, there has to exist a and b where starting at the differing bit, a and b have 0s and 1s that are complementary of each other --> XOR to 111...(from the differing bit to the end of bit string)
```c++
int maximizingXor(int l, int r) {
	// log2(l ^ r) gets the position of the greatest differing bit
	// shift 1 by log2(l ^ r)+1 so it's one more than the desired number
	// subtract one from it
	return (1 << (int(log2(l ^ r)) + 1)) - 1;
}
```

## Next Permutation
* Implement next permutation, which rearranges numbers into the lexicographically next greater permutation of numbers. (max would be digits in descending order)
* IDEA: from the back, find the first number that is ascending (the numbers after it are guaranteed to be in order), move the smallest number after it (but not equal to), swap them, and reverse the rest
* if already in descending order, reverse it so it's in the lowest possible order (i.e., sorted in ascending order).
```c++
void nextPermutation(vector<int>& nums) {
	int i = nums.size() - 2;
	// find the first element that is descending (from the back)
	// e.g. 5 in 1 2 5 8 7
	while (i >= 0 && nums[i + 1] <= nums[i]) {
		i--;
	}

	// Find the first element after i that is greater than nums[i]
	// e.g. 7 in 1 2 5 8 7
	if (i >= 0) {
		int j = nums.size() - 1;
		while (j >= 0 && nums[j] <= nums[i]) {
			j--;
		}

		// swap them, so now it's the greatest number starting with j, the part after j is still gauranteed to be descending since nums[j] <= nums[i]
		swap(nums, i, j);
	}

	// reverse the part after i to make it the smallest number starting with j, which is the next lexicographical order
	// intuition: move the smallest bigger number to the front and make the following in ascending order through reverse --> next slightly bigger permutation
	reverse(nums, i + 1, nums.size() - 1);
}

void reverse(vector<int>& nums, int lo, int hi) {
	while (lo < hi) {
		swap(nums, lo++, hi--);
	}
}

void swap(vector<int>& nums, int i, int j) {
	int tmp = nums[i];
	nums[i] = nums[j];
	nums[j] = tmp;
}
```
## Sansa and XOR (Hackerrank)
* Find the value obtained by XOR-ing the contiguous subarrays, then XOR those values to get one number
* XOR is associative
* Trick!! 
	* In an array whose size is even, every number appears even number of times --> XOR to 0
	* In an array whose size is odd, numbers with odd indices appear odd number of times --> XOR them to get the answer (test out with size = 3, 5)
```c++
int sansaXor(vector<int> arr) {
	int n = arr.size();
	if(n % 2 == 0) return 0;
	else {
		int ans=0;
		for(int i=0; i < n; i+=2) {
			ans = ans ^ arr[i];
		}
		return ans;
	}
}
```
## AND Product
* Given two integers, AND all numbers in between (inclusive)
* Trick: zero the all the bits from the first differing bit (to the right of it)
```c++
long andProduct(long a, long b) {
	// a ^ b to get where first differing bit string starts
	// + 1 to overshoot the bits we're zeroing
	// - 1 to get 00...11111
	// ~ negates
	// AND with either a or b gets the answer
	long firstDif = ~((1 << ((int)log2(a ^ b) + 1))-1) & a;
	return firstDif;
}
```
## N-Queens II
* recursion!!
```c++
int count =0;

bool isSafe(vector<vector<bool>>& mat , int row, int col, int n){
	//we are checking only upper parts because after this row (inclusive) there won't be any queens filled now
	// we are moving from bottom to top (current index and upward)
	// already know won't have overlapping queens in a row because ind represents row, which is unique to each queen --> only need to check if they overlap in the current column
	for(int i=row-1; i >= 0; i--) {
		if(mat[i][col] == true){
			return false;
		}
	}

	// for left upper diagonal check from the current position
	// HEED: && must be used for the middle condition, not a ,
	for(int i=row-1, j=col-1; i>=0 && j>=0; i--, j--) {
		if(mat[i][j]==true)
			return false;
	}

	// for right upper diagonal check from the current position
	for(int i=row-1, j=col+1; i >= 0 && j < n; i--, j++) {
		if(mat[i][j]==true)
			return false;
	}

	return true; 
}

void checkNQueens(vector<vector<bool>>& mat,int ind,int n){

	//it means we have checked all 0 to n-1 means total n rows and placed there the queen successfully so if ind==n then we found a solution so count++ and return 
	if(ind == n) {
		count++;
		return;
	}
	// n represents the side length of chess board and number of queens
	// ind also represents row index --> not repeating
	// i traverses columns
	for(int i=0; i < n; i++){
		// check if that spot can be placed a queen
		if(isSafe(mat,ind,i,n)){
			// mark it true and check for the possibility
			mat[ind][i] = true; 
			// ind + 1 means one more queen is checked
			checkNQueens(mat, ind+1, n);
			// once checked, mark it false again to check for the next column
			mat[ind][i] = false;
		}         
	}   
}

int totalNQueens(int n) {
	vector<vector<bool>> mat(n,vector<bool>(n,false));//initialize all places with false
	checkNQueens(mat,0,n);
	return count; 
}
```
## Find First and Last Position of Element in Sorted Array
* typical two pointers
* idea: find the entire interval once encounter a target number
```c++
vector<int> searchRange(vector<int>& nums, int target) {
	vector<int> ans(2, -1);
	int lo=0, mid, hi=nums.size()-1;
	while(lo <= hi) {
		mid = (lo+hi) / 2;
		if(nums[mid] > target) hi = mid-1;
		else if(nums[mid] < target) lo = mid+1;
		else {
			// once encounter one target number, get the interval and return here
			if(mid == 0 || nums[mid-1] != nums[mid]) {
				ans[0] = mid;
			}
			else {
				int tmp=mid;
				while(tmp != 0 && nums[tmp] == nums[tmp-1]) tmp--;
				ans[0] = tmp;
			}
			if(mid == nums.size()-1 || nums[mid+1] != nums[mid]) {
				ans[1] = mid;
			}
			else {
				int tmp=mid;
				while(tmp != nums.size()-1 && nums[tmp] == nums[tmp+1]) tmp++;
				ans[1] = tmp;
			}
			return ans;
		}   
	}
	// always return [-1, -1] here
	return ans; 
}
```
## Search in Rotated Sorted Array II
* There is an integer array nums sorted in non-decreasing order (not necessarily with distinct values). nums is right rotated by a number. Return true if target is in nums, or false if it is not in nums.
* Idea: 
	* normal binary search with tweaks
	* cut the vector in half --> one side is sorted, one is not
* Cheat: use std.find()
```c++
bool search(vector<int>& nums, int target) {
	int i = 0, j = nums.size()-1;
	while(i<=j){
		// not necessary
		while(i<j && nums[i]==nums[j] && nums[i]!=target) {
			i++;
			j--;
		}
		int mid = i + (j-i)/2;
		if(nums[mid] == target || nums[i] == target || nums[j] == target) return true;
		if(nums[mid] < target){
			if(nums[j] > target){
			// if target is presented in right half and right half is sorted, search in right half, could say if(nums[i] > target) cuz left unsorted implides right half is sorted
				i = mid+1;
			}
			// else not sure which side target is on, search the entire thing, but know i and j can’t be target, increment/decrement
			else{
				i++;
				j--;
			}    
		}
		else {
			if(nums[i] < target) {
			// if target is present in left half and left half sorted, search left half
				j=mid-1;
			}
			// else not sure which side target is on, search the entire thing, but know i and j can’t be target, increment/decrement
			else{
				i++;
				j--;
			}
		}
	}
	return false;
}
```
## Sort List
* Given the head of a linked list, return the list after sorting it in ascending order in O(nlogn)
* interesting recursion: two recursive functions, sortList is dependent on mergeTwoLists --> recursion within recursion!?
```c++
ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
	if (!l1) return l2;
	if (!l2) return l1;
	if (l1->val <= l2->val) {
		l1->next = mergeTwoLists(l1->next, l2);
		return l1;
	}
	else {
		l2->next = mergeTwoLists(l1, l2->next);
		return l2;
	}
}

ListNode* sortList(ListNode* head) {
	if(!head || !head->next) return head;

	ListNode *tail=head->next, *curr=head;
	// split the list into two even parts to optimize merge sort
	while(tail && tail->next) {
		curr=curr->next;
		tail=tail->next->next;
	}

	// tail becomes the head of the second half
	tail=curr->next;
	// terminate the first half
	curr->next=NULL;
	// breakdown each sub-lists down to one element, sort smaller pieces, then back up to bigger pieces
	return mergeTwoLists(sortList(head), sortList(tail));
}
```
## Cipher (Hackerrank)
* IMPORTANT XOR PROPERTY: if c = a xor b, then a = b xor c (also b = a xor c) --> three variables all interchangeable in the equation
* Every message is encoded to its binary representation. Then it is written down k times, shifted by k bits. Each of the columns is XORed together to get the final encoded string. Given the encoded string and k, solve for the original string.
* Use the cancelling property of XOR!!!
```c++
// assume k is 4
// res = p1, p2, p3, ...
string cipher(int k, string s) {
	int n=s.size()-k+1;
	int arr[n];
	string res;

	for(int i = 0; i < n; i++) {
		arr[i] = (int)(s[i] == '1');

		if(i==0)
			res+=to_string(arr[0]);
		else if(i < k)
			// e.g. res[2] = arr[2] ^ arr[1] = (p2 ^ p1 ^ p0) ^ (p1 ^ p0) = p2
			res+=to_string(arr[i]^arr[i-1]);
		else
			// e.g. res[4] = arr[4] ^ arr[3] ^ res[0] = (p1 ^ p2 ^ p3 ^ p4) ^ (p0 ^ p1 ^ p2 ^ p3) ^ p0 = p4
			res+=to_string(arr[i]^arr[i-1]^(res[i-k]-'0'));
	}
	return res;
}
```
## Xoring Ninja
* Sum all XORed number of all the power sets of the array
* Property: a+b = (a|b) + (a&b) and a^b = (a|b) - (a&b)
* Trick!! OR all elements then multiply by 2^(n-1) --> testing out with an array of size 2
```c++
long long xoringNinja(vector<int> arr) {
	long long r1 = 0, r2 = 1, m=1e9+7;
	int n = arr.size();
	for (int j(0); j < n; ++j) {
		r1 |= arr[j];
		r1 %= m;
		if (j)
			r2 <<=1;
		r2 %=m;
	}
	return (r1 * r2)%m;
}
```
## Number of Islands
```c++
// bfs
int numIslands(vector<vector<char>>& grid) {
	int m = grid.size(), n = grid[0].size();
	int ans = 0;
	queue<pair<int, int>> q;
	pair<int, int> dir[] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}}; 
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			if (grid[i][j] == '0') continue;
			ans++;
			grid[i][j] = '0';
			q.push({i, j});
			for (; !q.empty();) {
				auto [r, c] = q.front();
				q.pop();
				for (auto d : dir) {
					int nr = r + d.first, nc = c + d.second;
					if (nr >= 0 && nr < m && nc >= 0 && nc < n && grid[nr][nc] == '1') {
						grid[nr][nc] = '0';
						q.push({nr, nc});
					}
				}
			}
		}
	}
	return ans;
}
```
```c++
// dfs
int m, n;
pair<int, int> dir[4] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}}; 

void dfs (int r, int c, vector<vector<char>>& grid) {
	if (r < 0 || r >= m || c < 0 || c >= n || grid[r][c] == '0') return;
	grid[r][c] = '0';
	for (auto d : dir) 
	dfs (r + d.first, c + d.second, grid);
}

int numIslands(vector<vector<char>>& grid) {
	m = grid.size(), n = grid[0].size();
	int ans = 0;
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			if (grid[i][j] == '0') continue;
			ans++;
			dfs(i, j, grid);
		}
	}
	return ans;
}
```
## XOR from 1 to n
* Periodic function, list out cases to observe the pattern
```c++
int computeXOR(int n) {

	// If n is a multiple of 4
	if (n % 4 == 0)
		return n;

	// If n%4 gives remainder 1
	if (n % 4 == 1)
		return 1;

	// If n%4 gives remainder 2
	if (n % 4 == 2)
		return n + 1;

	// If n%4 gives remainder 3
	return 0;
}
```
## Xor-sequence (Hackerrank)
* A[L] ^ A[L+1] ^ A[L+2] ^ ... ^ A[R] = A[1] ^ A[2] ^ ... A[L-1] ^ (A[L] ^ A[L+1] ^ ... ^ A[R]) ^ A[1] ^ A[2] ^ ... ^ A[L-1]
	* front and back parts cancel because a ^ a = 0
* A[x] = 1 ^ 2 ^ 3 ^ ... ^ x. G(X) = A[1] ^ A[2] ^ ... ^ A[X]
* answer equals A[1] ^ A[2] ^ ... A[L-1] ^ (A[L] ^ A[L+1] ^ ... ^ A[R]) ^ A[1] ^ A[2] ^ ... ^ A[L-1], which is G(R)^G(L-1)
```c++
// observe the pattern by testing out simple functions
long long G(long long x){
	long long a = x % 8;
	if(a == 0 || a == 1) return x;
	if(a == 2 || a == 3) return 2;
	if(a == 4 || a == 5) return x+2;
	if(a == 6 || a == 7) return 0;
	return 0;
}

long xorSequence(long l, long r) {
	return G(r)^G(l-1);
}
```
## House Robber
* Given an integer array nums representing the amount of money of each house, return the maximum amount of money you can rob tonight without robbing two adjacent houses.
* DP!
```c++
// DP idea, but use prev, prevprev to keep track of previous terms
int rob(vector<int>& nums) {
	if(nums.size()==1)
	return nums[0];
	int curr;
	int prev=nums[0], prevprev=0;
	for(int i=1;i<nums.size();i++){
		curr=max(prevprev+nums[i], prev);
		prevprev=prev;
		prev=curr;
	}
	return curr;
}

// typical DP (bottom-up with memoization) with an array
int rob(vector<int>& A) {
	int n = A.size();
	if(n==1){
		return A[0];
	}
	if(n==2){
		return max(A[0],A[1]);
	}
	
	int dp[n];
	dp[0]=A[0];
	dp[1]=max(dp[0],A[1]);
	for(int i=2;i<n;i++){
		dp[i] = max(dp[i-1],dp[i-2]+A[i]);
	}
	return dp[n-1];
}

```
## Sum vs XOR (Hackerrank)
* Given an integer n, find the number of x such that 0 <= x <= n, x + n = x ^ n
* Trick! Raise 2 to the power of the number of zeroes in binary n
	* Each 0 can either stay 0 or be XORed to 1 and sum to the same number, 2 possibilities for each 0
```c++
long sumXor(long n) {
	long ans=1;
	while(n) {
		// if((n & 1) == 0) 
		// if n is odd don't multiply ans, if n is even multiply ans by 2
		ans <<= (n % 2? 0 : 1);
		n >>= 1;
	}
	return ans;
}
```
## Gas Station
* Use the fact that there exists only ONE unique solution or no solution
* There are n gas stations along a circular route, where the amount of gas at the ith station is gas[i]. You have a car with an unlimited gas tank and it costs cost[i] of gas to travel from the ith station to its next (i + 1)th station. You begin the journey with an empty tank at one of the gas stations.
* return the starting gas station's index if you can travel around the circuit once in the **clockwise** direction, otherwise return -1. If there exists a solution, it is ***guaranteed to be unique***.
```c++
int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
	int ans=0,cap=0,total=0,n = gas.size();
	// cap represents the remaining money at i, if reached negative, current and all the previous i's can't be where it starts, set i to the next index, cap to zero
	for(int i=0;i<n;i++) {
		cap += gas[i]-cost[i];
		if(cap < 0) {
			ans=i+1;
			total += cap;
			cap = 0;
		}
	}
	// total + cap is the total difference between gas and cost
	// if >= 0, must exist a station that works (it's a loop!!)
	// if < 0, no solution
	if(total + cap < 0) return -1;
	else return ans;
}
```
## The Great XOR (Hackerrank)
* Given a long integer *x*, count the number of values of *a* satisfying the following conditions:
	* a ^ x  > x
	* 0 < a < x
* Try out cases and observe the pattern --> sum of 0's raised to the power of their positions
	* can set any 0-bit to 1 and the positions below it can be either 1 or 0 and they XOR to > x
```c++
long theGreatXor(long x) {
	long ans=0, idx=0;
	while(x) {
		if((x & 1) == 0) {
			ans+= (long)1 << idx;
		}
		idx++;
		x >>= 1;
	}
	return ans;
}
```
## Kth Smallest Element in a BST
* Trick: use an index passed by reference --> changes its value throughout recursion
* Given the root of a binary search tree, and an integer *k*, return the kth **(1-indexed)** smallest element in the tree.
```c++
int kthSmallest(TreeNode* A, int k) {
	int res;
	inorder(A, k, res);
	return res;
}
// k is passed by reference!!!
// subtract 1's in inorder order
void inorder (TreeNode* A, int &k, int &res) {
	if (A == NULL)
	return;

	inorder(A->left, k, res);
	k--;
	if(k == 0) {
		res = A->val;
		return;
	}
	inorder(A->right, k, res);
}
```
## Lowest Common Ancestor of a Binary Tree
* Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree (**allow a node to be a descendant of itself**)
* Find the two nodes then traverse back up
```c++
TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
	if(!root) return nullptr;
	if(root->val==p->val || root->val==q->val) return root;
	// trace back from the roots
	TreeNode *l=lowestCommonAncestor(root->left,p,q);
	TreeNode *r=lowestCommonAncestor(root->right,p,q);
	if(l && r) return root;
	// take care of the case where p or q is a descendent of the other
	return l?l:r;
}
```
## Minimum Size Subarray Sum
* Given an array of positive integers nums and a positive integer target, return the minimal length of a contiguous subarray [numsl, numsl+1, ..., numsr-1, numsr] of which the sum is greater than or equal to target. If there is no such subarray, return 0 instead.
* Sliding window (cuz **contiguity** required) to achieve time complexity O(N)!
```c++
int minSubArrayLen(int target, vector<int>& nums) {
	int n=nums.size();
	int i=0, j=0, sum=0, count=INT_MAX;
	while(i < n) {
		sum+=nums[i];
		while(sum >= target) {
			count = min(count, i-j+1);
			sum-=nums[j];
			j++;
		}
		i++;
	}
	return (count==INT_MAX? 0 : count);
}
```
## Ice Cream Parlor (Hackerrank)
* Two friends like to pool their money and go to the ice cream parlor. They always choose two distinct flavors and they spend all of their money. Given a list of prices for the flavors of ice cream, select the two indices that will cost all of the money they have. 
* Idea: add indices to map while checking if the complement is in it
```c++
vector<int> icecreamParlor(int m, vector<int> arr) {
	map<int, int> mm;
	for(int i=0; i < arr.size(); i++) {
		// take care of duplicates
		int x = arr[i], y=m-x;
		if(mm[y] != 0) {
			int j=mm[y], k=i+1;
			return {j, k};
		}
		mm[x] = i+1;
	}
	return {};   
}
```
##  Different Ways to Add Parentheses
* Given a string expression of numbers and operators, return all possible results from computing all the different possible ways to group numbers and operators. You may return the answer in any order.
* Trick: memoization and recursion!
```c++
unordered_map<string, vector<int>> memo;
int calc(int a, int b, char& op){
	if(op=='+') {
		return a+b;
	}
	else if(op=='-') {
		return a-b;
	}
	else {
		return a*b;
	}
}
vector<int> diffWaysToCompute(string expression) {

	// keep memo table of indices for easy retrieval
	if(memo.find(expression)!=memo.end()){
		return memo[expression];
	}
	int n = expression.length();
	vector<int> ans;
	for(int i=0;i<n;++i){
		// break into left and right expression if s[i] is an operation
		if(expression[i]=='+'||expression[i]=='-'||expression[i]=='*') {
			vector<int> left = diffWaysToCompute(expression.substr(0,i));
			vector<int> right = diffWaysToCompute(expression.substr(i+1));
			for(int l:left){
				for(int r:right){
					ans.push_back(calc(l,r,expression[i]));
				}
			}
		}
	}
	// take care of the case of a single digit, push back the digit
	if(ans.empty()) {
		ans.push_back(stoi(expression));
	}
	
	// assosiate the expression with all possible outcomes
	memo[expression] = ans;
	return memo[expression];      
}
```
## Delete Node in a Linked List
* Write a function to delete a node in a singly-linked list. You will not be given access to the head of the list, instead you will be given access to the node to be deleted directly.
* Easy but tricky!!
	* Replace the current node value by the next node value, then skip the next node
```c++
void deleteNode(ListNode* node) {
	swap(node->val, node->next->val);
	node->next = node->next->next;
}
```
## Product of Array Except Self
* Given an integer array nums, return an array answer such that answer[i] is equal to the product of all the elements of nums except nums[i].
* O(N) and no division allowed!
```c++
vector<int> productExceptSelf(vector<int>& nums) {
	vector<int> ret(nums.size());
	ret[0] = 1; // left most is 1 bc no number to its left
	// on the way there, multiply numbers up to but not including itself (left hand side)
	for(int i=1; i < nums.size(); i++) {
		ret[i] = ret[i-1] * nums[i-1];
	}
	int right=1; // right most is 1 bc no number to its right
	// on the way back, multiply numbers to the right of the number (not including itself)
	for(int i=nums.size()-1; i >= 0; i--) {
		ret[i] = ret[i] * right;
		right *= nums[i];
	}
	return ret;
}
```
## Connected Cells in a Grid
* Given an  matrix, find and print the number of cells in the largest region (adjacent to each other horizontally, vertically, or diagonally) in the matrix. Note that there may be more than one region in the matrix.
```c++
// dfs
vector<int> posX = {1, 0, -1, 0, 1, -1, -1, 1};
vector<int> posY = {0, -1, 0, 1, -1, -1, 1, 1};
void dfs(vector<vector<int>>& v, int x, int y, int& cur) {
	if(x < 0 || x >= v.size() || y < 0 || y >= v[0].size() || v[x][y]==0) 
		return;
	// must be a one and in boundary here
	cur++;
	v[x][y] = 0;
	for(int i=0; i < posX.size(); i++) {
	// don't increment x, y in this for loop!!!
		dfs(v, x + posX[i], y + posY[i], cur);
	}
}

int connectedCell(vector<vector<int>> matrix) {
	int curMax=0;
	for(int i=0; i < matrix.size(); i++) {
		for(int j=0; j < matrix[0].size(); j++) {
			int cur=0;
			dfs(matrix, i, j, cur);
			if(cur > curMax) curMax = cur;
		}
	}
	return curMax;
}
```
## Search a 2D Matrix II
* Write an efficient algorithm that searches for a target value in an m x n integer matrix. The matrix has the following properties:
	* Integers in each row are sorted in ascending from left to right.
	* Integers in each column are sorted in ascending from top to bottom.
* O(n+m), about equally efficicent with small n + m
```c++
bool searchMatrix(vector<vector<int>>& mtx, int t) {
	int m = mtx.size(), n = mtx[0].size();
	int i = 0, j = n - 1; // starts on the greatest number for j, smallest number for i
	for (; i < m && j >= 0;) {
		// it's i in mtx[i][j] for if statement!! Can go down then go left
		if (mtx[i][j] > t) j--; // can't do i-- bc already starts on the lowest i side
		else if (mtx[i][j] < t) i++; // can't do j++ bc already starts on the greatest j side
		else return true;
	}
	return false;
}
```
## Palindrome Linked List
```c++
// with vector
bool isPalindrome(ListNode* head) {
	vector<int> v;
	while(head) {
		v.push_back(head->val);
		head = head->next;
	}
	int n=v.size();
	for(int i=0; i < n/2; i++) {
		if(v[i] != v[n-1-i]) return false;
	}
	return true;
}

// with stack
bool isPalindrome(ListNode* head) {
	stack<int> s;
	ListNode* p = head;
	while(head) {
		s.push(head->val);
		head = head->next;
	}
	while(!s.empty()) {
		if(s.top() != p->val) return false;
		p = p->next;
		s.pop();
	}
	return true;
}

// one pass, reverse half way then compare
ListNode *reverse(ListNode *head){
	ListNode *curr=head, *prev=NULL, *next=NULL;
	while(curr) {
		next=curr->next;
		curr->next=prev;
		prev=curr;
		curr=next;
	}
	return head=prev;
}
bool isPalindrome(ListNode* head) {
	ListNode *curr=head, *slow=head, *fast=head;
	// take care of both even/odd number of nodes
	while(fast->next && fast->next->next){
		slow=slow->next;
		fast=fast->next->next;
	}
	slow->next=reverse(slow->next);
	slow=slow->next;
	while(slow){
		if(slow->val!=curr->val) return 0;
		slow=slow->next;
		curr=curr->next;
	}
	return 1;
}
```
## (good graph question!) Cut the Tree (Hackerrank) 
* There is an undirected tree where each vertex is numbered from **1** to **n**, and each contains a data value. The sum of a tree is the sum of all its nodes' data values. If an edge is cut, two smaller trees are formed. The difference between two trees is the absolute value of the difference in their sums. Given a tree, determine which edge to cut so that the resulting trees have a minimal difference between them, then return that difference.
* recursion --> dfs
* sum branches and compare the difference
```c++
vector<bool> vis;
int total=0, ans=1e9;
// tree is the adjacency list
vector<vector<int>> tree;

int dfs(int u, vector<int>& data) {
	// below is the sum of nodes up to the current node u
	int below = data[u-1];
	vis[u]=true;
	for(int i=0;i<tree[u].size();i++) {
		// check if visited so don't sum repeatedly
		if(vis[tree[u][i]] == false)
			below += dfs(tree[u][i], data);
	}
	// check the current difference between two trees, hence -2*below
	if(abs(total - (2*below)) < ans) {
		ans = abs(total - (2*below));
	}
	return below;
}

int cutTheTree(vector<int> data, vector<vector<int>> edges) {
	int n=data.size();
	tree.resize(n+1);
	vis.resize(n+1,false);
	for(int i=0; i < n; i++) {
		total += data[i];
	}
	
	// edges are given in 1-based index of nodes, therefore tree size == n+1
	for(int i=0; i < n-1; i++) {
		int u=edges[i][0], v=edges[i][1];
		tree[u].push_back(v);
		tree[v].push_back(u);
	}

	// traverse from the first node
	dfs(1, data);
	return ans;
}
```
