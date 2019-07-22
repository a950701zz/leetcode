#include <vector>
#include <iostream>
#include <map>
#include <stdlib.h>
#include <algorithm>    // std::max
#include "string"
#include <set>
#include<queue>
#include <stack>
using namespace std;
typedef unsigned __int8 uint8_t;
typedef unsigned __int32 uint32_t;
//                                     1
//
//class Solution
//{
//public:
//	vector<int> twoSum(vector<int>& nums, int target)
//	{
//		vector<int> ans;
//		map<int, int> hashT;
//		int size = nums.size();
//		map<int, int> ::iterator it;
//		for (int i = 0; i < size; i++)
//		{
//			if ((it = hashT.find(target - nums[i])) != hashT.end())
//			{
//				ans.push_back(it->second);
//				ans.push_back(i);
//				return ans;
//			}
//			else
//				hashT.insert(map<int, int>::value_type(nums[i], i));
//		}
//	}
//};

//                                     3
//
//class Solution
//{
//public:
//	vector<int> twoSum(vector<int>& nums, int target)
//	{
//		vector<int> ans;
//		map<int, int> hashT;
//		int size = nums.size();
//		map<int, int> ::iterator it;
//		for (int i = 0; i < size; i++)
//		{
//			if ((it = hashT.find(target - nums[i])) != hashT.end())
//			{
//				ans.push_back(it->second);
//				ans.push_back(i);
//				return ans;
//			}
//			else
//				hashT.insert(map<int, int>::value_type(nums[i], i));
//		}
//	}
//};
//
//class Solution {
//public:
//	int lengthOfLongestSubstring(string s) {
//		int rst = 1;
//		if (s.length() == 0) return 0;
//		for (int i = 0; i < s.length(); i++)
//		{
//			for (int j = i + rst; j <= s.length()-1; j++)
//			{
//				string sub = s.substr(i, j-i+1);  //�ַ�������j+1-i
//				if (allUnique(sub) == true)
//				{
//					int a = sub.length();
//					rst = max(rst, j - i + 1);
//				}
//				else break;
//			}
//		}
//		return rst;
//	}
//	
//	
//	bool allUnique(string s)
//	{
//		for (int i = 0; i < s.length()-1; i++)
//		{
//			for (int j = i+1; j < s.length(); j++)
//			{
//				if (s[i] == s[j]) return false;
//			}
//		}
//		return true;
//	}
//};


//7
//class Solution {
//public:
//	int reverse(int x) {
//		int a, b; 
//		int  rst = 0;
//		while (x / 10 != 0)
//		{
//			int pop = x % 10;
//			if (rst > INT_MAX / 10 || (rst == INT_MAX / 10 && pop > 7)) return 0;
//			if (rst < INT_MIN / 10 || (rst == INT_MIN / 10 && pop < -8)) return 0;
//			rst = rst * 10 + pop;
//			x = x / 10;
//		}
//		if (rst > INT_MAX / 10 || (rst == INT_MAX / 10 && x % 10 > 7)) return 0;
//		if (rst < INT_MIN / 10 || (rst == INT_MIN / 10 && x % 10 < -8)) return 0;
//		rst = rst * 10 + x % 10;
//		return rst ;
//		
//	}
//};


//35 ��ͨ������
//class Solution {
//public:
//	int searchInsert(vector<int>& nums, int target) {
//		if (nums.size() == 1)
//		{
//			if (target <= nums[0]) return 0;
//			else
//				return 1;
//		}
//		if (nums.size() == 2)
//		{
//			if (target <= nums[0]) return 0;
//			else
//				if (target > nums[0] && target <= nums[1]) return 1;
//				else
//					return 2;
//		}
//		for (int i = 1; i <= nums.size()-2;i++)
//		{
//			if (target <= nums[i - 1])
//				return i - 1;
//			if (target > nums[i - 1] && target <= nums[i])
//				return i;
//		}
//		if (target <= nums[nums.size() - 1]) return nums.size() - 1;
//		else
//			return nums.size();
//	}
////};
//
//
//	//���ַ�����
//
//class Solution {
//	public:
//		int searchInsert(vector<int>& nums, int target) {
//
//			/*if (nums.size() == 0 || nums[0] > target)
//				return 0;
//			if (nums[nums.size() - 1] < target)
//				return nums.size();*/
//			int low = 0; int hight = nums.size() - 1;
//			int mid = 0;
//			while (low<=hight)
//			{
//				mid = (low + hight) / 2;
//				if (nums[mid]<target)
//				{
//					low = mid + 1;
//				}
//				else if (nums[mid]>target)
//				{
//					hight = mid - 1;
//				}
//				else
//					return mid;
//			}
//
//			return nums[mid] >= target ? mid : mid + 1;
//		}
//	};


//70��¥��
//class Solution {
//public:
//	int climbStairs(int n) {
//		if (n < 1) return 1;
//		vector<int> dp(n+100);
//		dp[0] = 1; dp[1] = 1;
//		for (int i = 2; i <=n; i++)
//		{
//			dp[i] = dp[i - 1] + dp[i - 2];
//		}
//		return dp[3];
//	}
//};

// 53
//class Solution {
//public:
//	int maxSubArray(vector<int>& nums) {  //maxsum ��ʾ�����±�Ϊi��Ԫ�ص��Ӵ���������
//		int ans = nums[0];
//		int maxsum = nums[0];
//		for (int i = 1; i < nums.size();i++)
//		{
//			maxsum = maxsum + nums[i] > nums[i] ? maxsum + nums[i] : nums[i];
//			ans = maxsum > ans ? maxsum : ans;
//		}
//		return ans;
//	}
//};

//     121 
//class Solution {
//public:
//	int maxProfit(vector<int>& prices) {
//		int minprice = min(prices[0],prices[1]);
//		int maxprofit = prices[1] - prices[0] > 0 ? prices[1] - prices[0] : 0;
//		for (int i = 2; i < prices.size();i++)
//		{
//			if (minprice > prices[i-1]) minprice = prices[i-1]; //ǰһ�����С�۸�
//			maxprofit = max(maxprofit, prices[i] - minprice );
//
//		}
//		return maxprofit<0?0:maxprofit;
//	}
//};


// 198
//class Solution {
//public:
//	int rob(vector<int>& nums) {
//		if(nums.size() == 1) return nums[0];
//		if (nums.size() == 2) return max(nums[0],nums[1]);
//		if (nums.size() == 3) return max(nums[0]+nums[2], nums[1]);
//		if (nums.size() == 4) return max(nums[0] + nums[2], nums[1]);
//		
//		int current3 = nums[2] + nums[0];
//		int current2 = nums[1];
//		int current1 = nums[0];
//		int current4 = max(current2 + nums[3], current1 + nums[3]);
//		int middle = current4;
//
//		for (int i = 4; i < nums.size();i++)
//		{
//			current1 = current2;
//			current2 = current3;
//			current3 = current4;
//			current4 = max(current2 + nums[i], current1 + nums[i]);
//		
//		}
//		return max(current4,current3);
//
//	}
//};


//764
//class Solution {
//public:
//	int minCostClimbingStairs(vector<int>& cost) {
//		// rst = min{dp[i],dp[i-1]}
//		int currenti = 0;
//		int currenti_1 = 0;
//		int rst = 0;
//		for (int i = 2; i <= cost.size() - 1; i++)
//		{
//			rst = min(currenti + cost[i - 1], currenti_1 + cost[i - 2]);
//			currenti_1 = currenti;
//			currenti = rst;
//		}
//		int index = cost.size()-1;
//		return min(currenti + cost[index], currenti_1 + cost[index - 1]);
//	}
//};

//   169
//class Solution {
//public:
//	int majorityElement(vector<int>& nums) {
//		int current = nums[0];
//		int count = 0;
//		for (int i = 0; i <= nums.size()- 1;i++)
//		{
//			if (count == 0)
//			{
//				current = nums[i];
//				count++;
//			}
//			else if (current == nums[i])
//			{
//				count++;
//			}
//			else if (current != nums[i])
//			{
//				count = count - 1;
//			}
//		}
//		return current;
//	}
//};

// ��ȫƽ����
//class Solution {
//public:
//	int numSquares(int n) {
//		if (n == 1) return 1;
//		if (n == 2) return 2;
//		if (n == 0) return 0;
//		vector<int> f(n+1,INT_MAX);  //
//		f[0] = 0; f[1] = 1; f[2] = 2;
//		for (int i = 3; i <= n; i++)
//		{
//			for (int j = sqrt(i); j>=1; j--)
//			{
//				if (i == j*j)
//				{
//					f[i] = 1; break;
//				}
//				else
//				f[i] = min(f[i], f[j*j] + f[i-j*j]);  //���û���f[i+j*j]�����f[i]+1
//			}
//		}
//		return f[n];
//	}
//};
//
//class Solution {
//public:
//	int maxProduct(vector<int>& nums) {
//		if (nums.size() == 1) return nums[0];
//		if (nums.size() == 2) return max(max( nums[0], nums[1]), nums[0] * nums[1] );
//		int current = max( nums[0] * nums[1], nums[1] );
//		int current_ = min(nums[0] * nums[1], nums[1]);
//		int previous = nums[0];
//		int previous_ = nums[0];
//		int rst = max(current, previous);
//		for (int i = 2; i <= nums.size() - 1; i++)
//		{
//			previous = current;
//			previous_ = current_;
//			current = max(max(previous*nums[i], nums[i]),previous_*nums[i]);
//			current_ = min(min(previous*nums[i], nums[i]), previous_*nums[i]);
//			if (rst < current) rst = current;
//		
//		}
//		return rst;
//	}
//};


////��ת����
//class Solution {
//public:
//	void rotate(vector<int>& nums, int k) {
//		int sum = nums.size();
//		vector <int> copyed(nums);
//		for (int i = 0; i <= nums.size() - 1;i++)
//		{
//			nums[(i + k)%sum] = copyed[i];
//		}
//	}
//};


//�ƶ���
//class Solution {
//public:
//	void moveZeroes(vector<int>& nums) {
//	 
//		int count = 0; int sizes = nums.size();
//		for (int i = 0; i <= sizes-1;i++)
//		{
//			if (nums[i]==0)
//			{
//				count++;
//			}
//			else
//			{
//				nums[i - count] = nums[i];
//			}
//		}
//		for (int i = 0; i < count;i++)
//		{
//			nums[sizes - i - 1] = 0;
//		}
//	}
//};

//�󽻼�
//class Solution {
//public:
//	vector<int> intersect(vector<int>& nums1, vector<int>& nums2) {
//		map<int, int> map1; int size1 = nums1.size();
//	    int size2 = nums2.size();
//		vector<int> result;
//		for (int i = 0; i <= size1 - 1; i++)
//		{
//			map1[nums1[i]]++;
//		}
//		
//
//		for (int i = 0; i <= size2 - 1; i++)
//		{
//			if (map1.count(nums2[i]) == 0||map1[nums2[i]]==0)
//				continue;
//			else
//			{
//				result.push_back(nums2[i]);
//				map1[nums2[i]]=0;
//			}
//		}
//		return result;
//	}
//
//	
//};
//
//class Solution {
//public:
//	bool searchMatrix(vector<vector<int>>& matrix, int target) {
//		//if (matrix == [[]]) return true;
//		if (matrix.size() == 0) return false;
//		if (target <= matrix[0][0]) return false;
//		int rst = 0;
//		for (int i = 1; i <= matrix.size() - 1; i++)
//		{
//			if (target < matrix[i][i])
//			{
//				rst = i;
//				break;
//			}
//			else
//				if (target > matrix[i][i]) {
//					rst = i;
//					continue;
//				}
//				else
//					if (target == matrix[i][i]) return true;
//		}
//		if (rst == matrix.size() - 1) return false;
//	
//		//��û�з��أ�������ֻ�ôӵ�i�к͵�i����ɸѡ
//		//��i��
//		
//		for (int i = 0; i <= rst; i++)
//		{
//			if (target < matrix[rst][i]) break;
//			if (target == matrix[rst][i]) return true;
//			if (target > matrix[rst][i]) continue;
//		}
//
//		for (int i = 0; i <= rst; i++)
//		{
//			if (target < matrix[i][rst]) break;
//			if (target == matrix[i][rst]) return true;
//			if (target > matrix[i][rst]) continue;
//		}
//		return false;
//
//	}
//};

//  �����
//class Solution {
//public:
//	string largestNumber(vector<int>& nums) {
//		BubbleSort(nums); string rst;
//		for (int i = 0; i <= nums.size() - 1;i++)
//		{
//			rst += to_string(nums[i]);
//		}
//		if (rst[0] == '0') return "0";
//		return rst;
//	}
//	void BubbleSort(vector<int> &nums)
//	{
//		for(int i = 0; i < nums.size()-1; i++)
//		{
//			for (int j = 0; j <= nums.size()-2- i;j++)
//			{
//				if (isminus(nums[j], nums[j + 1]))
//				{
//					int temp = nums[j+1];
//					nums[j + 1] = nums[j];
//					nums[j] = temp;
//				}
//			}
//		}
//	}
//	bool isminus( int a, int b )
//	{
//		return to_string(a) + to_string(b) < to_string(b) + to_string(a);
//	}
//};

//ֻ����һ�ε���
//class Solution {
//public:
//	int singleNumber(vector<int>& nums) {
//		int rst = 0;
//		for (int i = 0; i < nums.size(); i++)
//			rst ^= nums[i];
//		return rst;
//	}
//};

//���Ʊ�����ʱ��2
//class Solution {
//public:
//	int maxProfit(vector<int>& prices) {
//		int profit = 0;
//		for (int i = 0; i < prices.size() - 1; i++)
//		{
//			profit += prices[i + 1] - prices[i]>0 ? prices[i + 1] - prices[i] : 0;
//		}
//		return profit;
//	}
//};

//�ַ�����
//class Solution {
//public:
//	int findContentChildren(vector<int>& g, vector<int>& s) {
//		sort(g.begin(), g.end()); int count = 0;
//		sort(s.begin(), s.end()); int i = 0; int j = 0;
//		if (g.size() == 0 || s.size() == 0) return 0;
//
//		for (; i <= g.size() - 1 && j <= s.size() - 1;)
//		{
//			if (s[j] >= g[i])
//			{
//				count++; i++; j++;
//			}//��ǰ���ɿ��Էַ�
//			else
//				j++;
//		}
//		return count;
//	}
//};

//������С��
//class Solution {
//public:
//	string fractionToDecimal(int numerator, int denominator) {
//		long long t = numerator, d = denominator;
//		map<long  long,int > A;  //����һ��Map A��¼���ǵ�ǰ�ĳ����Լ����������ǵ�ʱans��size,Ϊ������һ�γ���
//		                        //ͬ���ĳ���ʱ�Ὺʼѭ������¼����sizeȷ���������λ�á�
//		string ans;
//		if (t*d < 0) ans = "-";
//		t = abs(t); d = abs(d);
//		ans += to_string(t / d);  //��������
//		t = t%d;  //С������ȡ����
//		if (!t) return ans;
//		ans += ".";
//		while (t)   //������Ϊ0��ʱ��ѭ������ t�ǵ�ǰ���Է�ĸ����
//		{
//			if (A.count(t))  //���ظ��ĳ���������ֹͣѭ��
//			{
//				ans.insert(A[t], "(");
//				ans.push_back(')');
//				return ans;
//			}
//			A[t] = ans.size();
//			ans += '0' + t * 10 / d;
//			t = t * 10 % d;
//		}
//		return ans;
//	}
//};

//ȱʧ������
//class Solution {
//public:
//	int missingNumber(vector<int>& nums) {
//		int sum = nums.size();
//		for (int i = 0; i <= nums.size() - 1; i++)
//		{
//			sum ^= nums[i];
//			sum ^= i;
//		}
//		return sum;
//		
//	}
//};

//λ1�ĸ���   һ����������ȥ1����õ��Ľ���ǰ������ұߵ�1���0
//class Solution {
//public:
//	int hammingWeight(uint32_t n) {
//		int count = 0;
//		while (n)
//		{
//			n &= (n - 1);
//			count++;
//		}
//		return count;
//	}
//};


//�׳˺��0
//class Solution {
//public:
//	int trailingZeroes(int n) {
//		long long x = n; int i = 1;
//		long long y = 5, num = 0;
//		while (y <= x)
//		{
//			num++;
//			i++;
//			y = i * 5;
//		}
//		return num;
//	}
//};

//ʢˮ��������
//class Solution {
//public:
//	int maxArea(vector<int>& height) {
//		int i = 0, j = height.size() - 1; int result = 0;
//		while (i<j)
//		{
//			if (height[i] < height[j])
//			{
//				result = max(result, (j - i)*height[i]);
//				i++;
//			}
//			else
//			{
//				result = max(result, (j - i)*height[j]);
//				j--;
//			}
//		}
//		return result;
//	}
//};


//���ͬǰ׺
//class Solution {
//public:
//	string longestCommonPrefix(vector<string>& strs) {
//		if (strs.size() == 1) return strs[0];
//		if (strs.size() == 0) return "";
//		int mark = 0; int pol = 1;
//		for (int i = 0; i < strs.size() - 1; i++)
//			if (strs[i].size() == 0) return "";
//		for (int i = 0; i <= strs[0].size() - 1; i++)
//		{
//			for (int j = 1; j <= strs.size() - 1;j++)
//			{
//				
//				if (strs[0][i]!=strs[j][i])   
//				{
//					pol = -1;
//					mark = i;
//					goto here;
//				}
//			}
//			mark = i;
//		}
//
//		here:
//		if (mark == 0&&pol<0 )
//			return "";
//		else if (pol < 0) return strs[0].substr(0, mark);
//			else 
//			return strs[0].substr(0, mark+1);
//	}
//};
//
//class Solution {
//public:
//	vector<int> productExceptSelf(vector<int>& nums) {
//		vector <int> res(nums.size(), 1);
//		int num = 1;
//		for (int i = nums.size() - 1; i >= 1;i--)  //res�������ȴ洢��ǰλ���ұ����ĳ˻�
//		{
//			res[i - 1] = num * nums[i];
//			num *= nums[i];
//		}
//		num = 1;
//		for (int i = 0; i <= nums.size() - 1;i++)
//		{
//			res[i] *= num;
//			num *= nums[i];
//		}
//		return res;
//	}
//};

//����ˮ
//class Solution {
//public:
//	int trap(vector<int>& height) {
//		if (height.size() == 1 || height.size() == 0) return 0;
//		int MaxNum = 0; int index = 0;
//		for (int i = 0; i < height.size(); i++)
//		{
//			if (height[i] > MaxNum)
//			{
//				MaxNum = height[i];
//				index = i;
//			}
//		}
//		int last = height[0]; int drop = 0;
//		for (int i = 1; i < index; i++)
//		{
//			
//			if (height[i] < last)  //��һ�����Ӹ߶Ƚ���
//			{
//				drop += last - height[i];
//			}
//			if (height[i] >= last)
//			{
//				last = height[i]; continue;
//			}
//		}
//		last = height[height.size() - 1];
//		for (int i = height.size() - 2; i > index;i--)
//		{
//			if (height[i] < last)
//			{
//				drop += last - height[i];
//			}
//			if (height[i] >= last)
//			{
//				last = height[i]; continue;
//			}
//		}
//		return drop;
//	}
//};


//Ӳ������
//class Solution {
//public:
//	int coinChange(vector<int>& coins, int amount) {
//		vector<int>dp(amount + 1, amount + 1);
//		dp[0] = 0;
//		if (coins.size() == 0) return 0;
//		for (int i = 1; i <= amount;i++)
//		{
//			for (int j = 0; j <= coins.size() - 1; j++)
//			{
//				if (i >= coins[j])    //i�ǵ�ǰ��̬�滮��״̬������coins[j]ÿ��Ӳ��
//					dp[i] = min(dp[i], dp[i - coins[j]] + 1);
//				else
//					continue;
//			}
//		}
//		return dp[amount] == amount + 1 ? -1:dp[amount];
//	}
//};
//
//class Solution {
//public:
//	int longestConsecutive(vector<int>& nums) {
//		map<int, int> x; int rst = 1;
//		if (nums.size() == 0)return 0;
//		for (int i = 0; i < nums.size();i++)
//		{
//			int num = x.count(nums[i]);
//			if (num == 0)  //��һ�α���ʱ
//			{
//				if (x.count(nums[i]-1)!=0)//���������
//				{
//					if (x.count(nums[i]+1)!=0)  //���ұ���������
//					{
//						int newsequence = x[nums[i] - 1] + x[nums[i] + 1] + 1;
//						cout << (nums[i] - 1);
//						cout << x.count(nums[i] - 1);
//						x[nums[i] - x[nums[i] - 1]] = newsequence;
//						x[nums[i] + x[nums[i] + 1]] = newsequence;
//						x[nums[i] ] = newsequence;
//						rst = max(newsequence, rst);
//					}
//					else  //��������ڣ��ұ߲�����
//					{
//						int newsequence = x[nums[i] - 1] + 1;
//						x[nums[i] - x[nums[i] - 1]] = newsequence;
//						x[nums[i]] = newsequence;
//						rst = max(newsequence, rst);
//					}
//				}
//				else   //�����������
//				{
//					if (x.count(nums[i] + 1)!= 0) //�����������,�ұ�������
//					{
//						int newsequence = x[nums[i] + 1] + 1;
//						x[nums[i] + x[nums[i]+1]] = newsequence;
//						x[nums[i]] = newsequence;
//						rst = max(newsequence, rst);
//					}
//					else  //���Ҷ�������
//						x[nums[i]] = 1;
//				}
//			}
//		}
//		return rst;
//	}
//};


//������С��������
//class Solution {
//public:
//	int minSubArrayLen(int s, vector<int>& nums) {
//		int left = 0, right = 0, sumlen = 0, rst = INT_MAX;
//		if (nums.size() == 0) return 0;
//		bool stopflag = false;
//		while (true)
//		{
//			  //��ǰ��
//			if (sumlen>s && right!=nums.size())  //����s,�󻬴����ƶ�
//			{
//				rst = min(rst, right - left);
//				sumlen -= nums[left];
//				left++;
//			}
//			else if (sumlen == s && right != nums.size() )
//			{
//				rst = min(rst, right - left);
//				sumlen += nums[right];
//				right++;
//			}
//			else if (sumlen<s && right != nums.size())
//			{
//				sumlen += nums[right];
//				right++;
//			}
//			else  //�һ���ָ���˱߽�
//			{
//				if (sumlen < s)
//				{
//					break;
//				}
//				else
//					if (sumlen == 2) rst = min(rst, right - left);
//					else
//				{
//					rst = min(rst, right - left);
//					sumlen -= nums[left];
//					left++;
//					continue;
//				}
//					
//			}
//		}
//		return rst == INT_MAX ? 0:rst;
//	}
//};
//
//class Solution {
//public:
//	bool increasingTriplet(vector<int>& nums) {
//		if (nums.size() <= 2) return false; int minnum = nums[0]; int nextmin =INT_MAX;
//		for (int i = 1; i < nums.size(); i++)
//		{
//			if (nums[i] < minnum){
//				nextmin = minnum;
//				minnum = nums[i];
//			}
//			else if (nums[i]<=nextmin)
//			{
//				nextmin = nums[i];
//			}
//			else  //nums[i]>nextmin
//			{
//				if (nextmin!=minnum)
//				return true;
//				else {
//					nextmin = nums[i];
//				}
//			}
//		}
//		return false;
//	}
//};
//
//class Solution {
//public:
//	void rotate(vector<vector<int>>& matrix) {
//		int roundtimes = matrix.size() / 2;
//		for (int i = 0; i < roundtimes;i++)
//		{
//			int maxvalue = matrix.size() - 1 - i;
//			for (int j = i; j < matrix.size();j++)  // [i][j]
//			{
//				int roundindex = matrix.size() - 1 - 2 * i;
//				int temp = matrix[i][j + roundindex];  //ÿһ����λ�������� matrix.size()-1-2*i 
//				matrix[i][j + roundindex] = matrix[i][j];
//				matrix[i][j] = matrix[i + roundindex][j];
//				matrix[i + roundindex][j] = matrix[i + roundindex][j + roundindex];
//				matrix[i + roundindex][j + roundindex] = temp;
//			}
//		}
//	}
//};

//class Solution {
//public:
//	bool isPalindrome(string s) {
//		if (s.size() == 0) return true;
//		int head = -1, tail = s.size();
//		while (head < tail)
//		{
//			do 
//			{
//				head++; if (head >= s.size()) return true;
//				
//			} while (!(s[head] >= 'a'&&s[head] <= 'z' || s[head] >= 'A'&&s[head] <= 'Z'||s[head]>='0'&&s[head]<='9')); //head����ĸʱ
//			do
//			{
//				tail--; if (tail<=0) return true;
//			} while (!(s[tail] >= 'a'&&s[tail] <= 'z' || s[tail] >= 'A'&&s[tail] <= 'Z' || s[tail] >= '0'&&s[tail] <= '9'));
//			if (s[head] == s[tail])
//				continue;
//			if (s[head] >= '0'&&s[head] <= '9') return false;
//			if (s[tail] >= '0'&&s[tail] <= '9') return false;
//			if (abs(s[head] - s[tail]) == 32) return true;
//			else return false;
//		}
//		return true;
//	}
//};

//class Solution {
//public:
//	int countsum(int n)
//	{
//		int total = 0;
//		while (n != 0)
//		{
//			total += pow(n % 10, 2);
//			n = n / 10;
//		}
//		return total;
//	}
//	bool isHappy(int n) {
//		set<int> s;
//		int current = n % 10;
//		while (s.find(n)==s.end())  //set��û�е�ǰ����
//		{
//			s.insert(n);
//			if (countsum(n) == 1) return true;
//			else
//				n = countsum(n);
//		}
//		return false;
//	}
//};

//
//class Solution {
//public:
//	int findPeakElement(vector<int>& nums) {  //���ַ��������ַ�������ָ��������ʱ����ô�����м�ڵ�����ߵ���
//		int n = nums.size();
//		if (n == 1) return 0;
//		if (n == 2) if (nums[1] > nums[0]) return 1; else return 0;
//		if (nums[n / 2] > nums[n / 2 + 1] && nums[n / 2] > nums[n / 2 - 1])
//			return n / 2;
//		if (nums[n / 2] < nums[n / 2 + 1])  //���ұ�Ѱ��
//		{
//			for (int i = n / 2 + 1; i < n - 1; i++)
//			{
//				if (nums[i + 1] < nums[i]) return i;
//			}
//			return n - 1;
//		}
//
//		for (int i = n / 2 - 1; i >= 1; i--)
//		{
//			if (nums[i - 1] < nums[i]) return i;
//		}
//		return 0;
//	}
//};

//class Solution {
//public:
//	uint32_t reverseBits(uint32_t n) {
//		uint32_t ans = 0;
//		//���Ƶı���
//		int i = 32;
//		while (i--)
//		{
//			ans <<= 1;
//			ans += n & 1;
//			n >>= 1;
//		}
//		return ans;
//	}
//};

//struct Point {
//	int x;
//	int y;
//	Point() : x(0), y(0) {}
//	Point(int a, int b) : x(a), y(b) {}
//	
//};
//
//class Solution {
//public:
//	int maxPoints(vector<Point>& points) {    //ÿһ�α���һ���㣬ͨ��һ���һ��б��ȷ��Ψһ��ֱ�ߣ�һ�α����ҵ�����ֱ��ͨ���õ�����������������е������õ������Ž⡣
//		int res = 0;
//		for (int i = 0; i < points.size(); ++i) {
//			map<pair<int, int>, int> m;
//			int duplicate = 1;
//			for (int j = i + 1; j < points.size(); ++j) {
//				if (points[i].x == points[j].x && points[i].y == points[j].y) {   //�������һ������ͬһ���㣬�����ͬһ����Ļ�Ҫ��¼��Ҳ����һ��ֱ���ϡ�
//					++duplicate; continue;
//				}
//				int dx = points[j].x - points[i].x;
//				int dy = points[j].y - points[i].y;
//				int d = gcd(dx, dy);
//				++m[{dx / d, dy / d}];
//			}
//			res = max(res, duplicate);
//			for (auto it = m.begin(); it != m.end(); ++it) {
//				res = max(res, it->second + duplicate);
//			}
//		}
//		return res;
//	}
//	int gcd(int a, int b) {
//		return (b == 0) ? a : gcd(b, a % b); //�����Լ�����㷨
//	}
//};

//class Solution {
//public:
//	int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
//		queue<string> q;
//		map<string, int> m1; //���������е��ַ������ڲ����Ƿ����
//		map<string, int> re; //������
//		int n = wordList.size();
//		for (int i = 0; i < n; i++)
//			m1[wordList[i]] = 1;
//		re[beginWord] = 1;
//		q.push(beginWord);
//		while ((!q.empty()) && m1.size())
//		{
//			string now = q.front();
//			q.pop();
//			int num = re[now];
//			int llen = now.size();
//			for (int i = 0; i < llen; i++)
//			{
//				string temp = now;
//				for (char c = 'a'; c <= 'z'; c++)
//				{
//					if (temp[i] == c)
//						continue;
//					else
//						temp[i] = c;
//					if (m1.find(temp) != m1.end())
//					{
//						if (temp == endWord)
//							return num + 1;
//						q.push(temp);
//						re[temp] = num + 1;
//						m1.erase(temp);
//					}
//				}
//			}
//		}
//		return 0;
//	}
//};
//class Solution {
//public:
//	void reverseString(vector<char>& s) {
//		int j = s.size() - 1;
//		int i = 0;
//		while (i<j)
//		{
//			char x = s[i];
//			s[i] = s[j];
//			s[j] = x;
//			i++; j--;
//		}
//	}
//};

//const int MAXN = 26;
//class Trie{
//public:
//	bool is_str;  //�жϵ�ǰ�ڵ��Ƿ�Ϊһ���������ַ���
//	Trie *next[MAXN];
//	Trie()
//	{
//		is_str = NULL;
//		memset(next, 0, sizeof(next));
//	}
//
//	void insert(string word)
//	{
//		Trie *cur = this;
//		for (char w:word)
//		{
//			if (cur->next[w-'a']==NULL)
//			{
//				Trie *new_node = new Trie();
//				cur->next[w - 'a'] = new_node;
//			}
//			cur = cur->next[w - 'a'];
//		}
//		cur->is_str = true;
//	}
//	bool search(string word) {
//		Trie *cur = this;
//		for (char w : word){
//			if (cur != NULL)
//				cur = cur->next[w - 'a']; // ����curָ���ָ��ʹ��ָ����һ�����
//		}
//		return (cur != NULL&&cur->is_str); // curָ�벻Ϊ����curָ��ָ��Ľ��Ϊһ���������ַ�������ɹ��ҵ��ַ���
//	}
//
//	/** Returns if there is any word in the trie that starts with the given prefix. */
//	bool startsWith(string prefix) {
//		Trie *cur = this;
//		for (char w : prefix){
//			if (cur != NULL)
//				cur = cur->next[w - 'a'];
//		}
//		return (cur != NULL); // ���search(),����ֻ���ж�curָ���Ƿ�Ϊ�վ�����
//	}
//};

//class Solution {
//public:
//	int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
//		int len = beginWord.length();
//		map<string, vector<string>> allComboDict;
//		map<string, bool> visited;
//		for (int j = 0; j < wordList.size();j++)
//		{
//			string word = wordList[j];
//			for (int i = 0; i < len; i++)
//			{
//				string newword = word.substr(0, i) + "*" + word.substr(i + 1, len - (i + 1));
//				allComboDict[word].push_back(newword);
//			}
//			visited[word]= false;
//		}
//		//Queue for BFS
//		queue<pair<string, int>> Q;
//		pair<string, int> a (beginWord, 1);
//		Q.push(pair<string, int> (beginWord, 1));
//		//a visited map
//		
//		visited[beginWord] = true;
//		while (!Q.empty())   //BFS��ֹѭ������������Ϊ��
//		{
//			pair<string, int>node = Q.front();
//			Q.pop();
//			string currentword = node.first;  //�ڵ��string
//			int level = node.second;   //�ڵ��level
//			for (int i = 0; i < len; i++)
//			{
//				string intermediate = currentword.substr(0, i) + "*" + currentword.substr(i + 1, len - (i + 1));
//				map<string, vector<string>>::iterator it = allComboDict.begin();
//				while (it != allComboDict.end())
//				{
//					for (int j = 0; j < (it->second).size();j++)
//					{
//						string intermediate1 = it->second[j];
//						if (intermediate == intermediate1)  //�ҵ�����ͨ���ֶ�
//						{
//							if (it->first == endWord)
//								return level + 1;
//							else
//								if (visited[it->first] == false)  //û���ʹ�
//								{
//									visited[it->first] = true;
//									Q.push(pair<string,int>(it->first, level + 1));
//								}
//							continue;
//						}
//						
//					}
//					it++;
//				}
//			}
//		}
//		//BFS
//		return 0;
//	} 
//
//};

//class Solution {
//public:
//	bool isAnagram(string s, string t) {
//		sort(s.begin(), s.end());
//		sort(t.begin(), t.end());
//		return s == t ? true : false;
//	}
//};


//
//const int Maxn = 26;
//class Trie {
//public:
//	/** Initialize your data structure here. */
//	bool is_str;//��ʶ��ǰ�ڵ��Ƿ���һ���������ַ���
//	Trie *next[Maxn];  //ָ��Trie��ָ������
//	Trie() {
//		is_str = NULL;
//		memset(next, 0, sizeof(next)); //��ʼ��ָ������
//	}
//
//	/** Inserts a word into the trie. */
//	void insert(string word) {
//		Trie *cur = this; //����ָ��ǰ�ڵ�
//		for (char w : word)
//		{
//			if (cur->next[w - 'a'] == NULL)  //��ǰ�ڵ�û��ָ��ǰ�ַ�w�ķ�֧��Ҫ�½��ڵ�
//			{
//				Trie *new_node = new Trie();
//				cur->next[w - 'a'] = new_node;
//			}
//			cur = cur->next[w - 'a'];//ÿһ�ζ������һ����֧���ڵ�ָ����һ���ַ�
//		}
//		cur->is_str = true;//ָ����ĩβ���ַ��ˣ���ǰ�ڵ㴦��һ�������ĵ���
//	}
//
//	/** Returns if the word is in the trie. */
//	bool search(string word) {
//		Trie *cur = this;
//		for (char w : word)
//		{
//			if (cur->next[w - 'a'] == NULL)
//			{
//				return false;
//			}
//			cur = cur->next[w - 'a'];
//		}
//		return (cur->is_str);
//	}
//
//	/** Returns if there is any word in the trie that starts with the given prefix. */
//	bool startsWith(string prefix) {
//		Trie * cur = this;
//		for (char w : prefix)
//		{
//			if (cur->next[w - 'a'] == NULL)
//				return false;
//			cur = cur->next[w - 'a'];
//		}
//		return (cur != NULL);
//	}
//};

//class Solution {
//public:
//	int titleToNumber(string s) {
//		int len = s.length(); int rst = 0;
//		for (int i = 0; i < s.length();i++)
//		{
//			char a= s[i];
//			switch (a)
//			{
//			case 'A': rst += 1 * pow(26, len - i - 1); break;
//			case 'B': rst += 2 * pow(26, len - i - 1); break;
//			case 'C': rst += 3 * pow(26, len - i - 1); break;
//			case 'D': rst += 4 * pow(26, len - i - 1); break;
//			case 'E': rst += 5 * pow(26, len - i - 1); break;
//			case 'F': rst += 6 * pow(26, len - i - 1); break;
//			case 'G': rst += 7 * pow(26, len - i - 1); break;
//			case 'H': rst += 8 * pow(26, len - i - 1); break;
//			case 'I': rst += 9 * pow(26, len - i - 1); break;
//			case 'J': rst += 10 * pow(26, len - i - 1); break;
//			case 'K': rst += 11 * pow(26, len - i - 1); break;
//			case 'L': rst += 12 * pow(26, len - i - 1); break;
//			case 'M': rst += 13 * pow(26, len - i - 1); break;
//			case 'N': rst += 14 * pow(26, len - i - 1); break;
//			case 'O': rst += 15 * pow(26, len - i - 1); break;
//			case 'P': rst += 16 * pow(26, len - i - 1); break;
//			case 'Q': rst += 17 * pow(26, len - i - 1); break;
//			case 'R': rst += 18 * pow(26, len - i - 1); break;
//			case 'S': rst += 19 * pow(26, len - i - 1); break;
//			case 'T': rst += 20 * pow(26, len - i - 1); break;
//			case 'U': rst += 21 * pow(26, len - i - 1); break;
//			case 'V': rst += 22 * pow(26, len - i - 1); break;
//			case 'W': rst += 23 * pow(26, len - i - 1); break;
//			case 'X': rst += 24 * pow(26, len - i - 1); break;
//			case 'Y': rst += 25 * pow(26, len - i - 1); break;
//			case 'Z': rst += 26 * pow(26, len - i - 1); break;
//			default:
//				break;
//			}
//		}
//		return rst;
//	}
//};

//struct BSTNode{
//	int val;  //��ǰֵ
//	int count;  //��nums�б�valС�����ĸ���
//	BSTNode *left; //������ָ��
//	BSTNode *right; // ������ָ��
//	BSTNode(int x) :val(x), left(NULL), right(NULL), count(0){};//��Ҫά��һ��count����Ϊ����ʱ����
//	//�Ὣÿ��������������Ҫ��¼ÿһ���ڵ��countֵ
//};
//
//void BST_insert(BSTNode *node, BSTNode *insert_node, int &count_small)
//{
//	//��node�ڵ㿪ʼ�����µ�insert_node�ڵ�
//	if (node->val >= insert_node->val)    //�½ڵ�С�ڵ��ھɽڵ�
//	{
//		node->count++;  //�ȵ�ǰ�ڵ�С�Ľڵ����Ҫ����
//		if (node->left)   //�½ڵ���뵽���
//			BST_insert(node->left, insert_node, count_small);
//		else
//			node->left = insert_node;
//		//����Ľڵ�ȵ�ǰ�ڵ�ҪС��
//	}
//	else
//	{
//		count_small += node->count + 1;
//		if (node->right)  //��ǰ�ڵ����������Ϊ��
//			BST_insert(node->right, insert_node, count_small);
//		else
//			//�������ǿ�
//			node->right = insert_node;
//	}
//}
//
//class Solution{
//public:
//	vector<int> countSmaller(vector<int> &nums)
//	{
//		int n = nums.size();
//		if (n == 0) return{}; //���ؿյ�ʱ�򷵻ص���һ�����飬�ô����ű�ʾ
//		vector<int>  count;  // �����������vector
//		count.push_back(0); // ���һ��Ԫ��ֵΪ0
//		BSTNode *node = new BSTNode(nums[n - 1]);  //����n-1���ڵ�,node��valΪnums[n-1]�Ľڵ�
//		int count_small;
//		for (int i = 1; i < n; i++)
//		{
//			count_small = 0; //ÿһ�ζ���nums[n-1]�Ľڵ㿪ʼ�����µĽڵ�
//			BST_insert(node, new BSTNode(nums[n - i - 1]), count_small);//����nums[n-i-1]��Ԫ�ص���������
//			count.push_back(count_small);
//		}
//		delete node;
//		reverse(count.begin(), count.end());
//		return count;
//	}
//};

//class Solution {
//public:
//	int findDuplicate(vector<int>& nums) {
//		sort(nums.begin(), nums.end());
//		for (int i = 0; i < nums.size()-1;i++)
//		{
//			if (nums[i]==nums[i+1])
//			{
//				return nums[i];
//			}
//		}
//		return -1;
//	}
//};

//class Solution {
//public:
//	//����ָ��Ļ���˼���������ָ��֮��ʼ��Ϊ��ָ�����ߵ�·�̣���ָ�������Ժ���ָ��һ�������������Ļ�����
//	// ��Ϊ��ʼ����m����m�Ǵ���㵽���ľ��룬���Դ�ʱ��ָ������m���͵��˻������
//	// ��ʱ�ÿ�ָ���ͷ�ߣ���m����Ҳ������㣬��ָ�������������Ҳ�����ظ�������
//	int findDuplicate(vector<int>& nums) {
//		int fast = nums[0];
//		int slow = nums[0];  //��nums[0]������ʼλ�ã���ָ������ʱҪ�ص�nums[0]
//		do 
//		{
//			slow = nums[slow];  //��һ��
//			fast = nums[nums[fast]]; //��ǰ������
//		} while (fast!=slow);
//
//		// �������ָ��ص����
//		fast = nums[0];
//		while (fast != slow)  //����Ѿ���������ôֱ�ӷ��أ�������do while
//		{
//			fast = nums[fast];
//			slow = nums[slow];
//		} 
//
//		return fast;
//	}
//};
//class Solution {
//public:
//	int numIslands(vector<vector<char>>& grid) {
//		if (grid.size()==0)
//			return 0;
//		vector<vector<bool>> visited(grid.size(), vector<bool>(grid[0].size(), false));
//		int x = grid.size(), y = grid[0].size(); int res = 0;
//		for (int i = 0; i < x;i++)
//		{
//			for (int j = 0; j < y;j++)
//			{
//				if (grid[i][j]=='1')
//				{
//					res++; BFS(grid, i, j, visited);
//				}
//			}
//		}
//		return res;
//	}
//
//	void BFS(vector<vector<char>>& grid, int xx, int yy, vector<vector<bool>>& visited)
//	{
//		int x = grid.size(); int y = grid[0].size();
//		//x�ǵ�һ��vector y�ǵڶ���vector��index
//		if (xx < 0 || xx >= x || yy < 0 || yy >= y)  //����Խ�磬����
//			return;
//		else
//			if (visited[xx][yy] == true || grid[xx][yy] == '0')  //����������
//				return;
//		  else
//		{
//			grid[xx][yy] = '0';
//			visited[xx][yy] = true;//����Ϊ���ʹ�
//			BFS(grid, xx + 1, yy, visited);
//			BFS(grid, xx - 1, yy, visited);
//			BFS(grid, xx , yy + 1, visited);
//			BFS(grid, xx , yy - 1, visited);
//		}
//	}
//};
//class Solution {
//public:  //�����зֿ����������Խ��ײ��ˡ���Ҫ���������ע��ŷ� �����ظ�����
//	void wiggleSort(vector<int>& nums) {
//		vector<int> copyed(nums);
//		if (nums.size() == 0) return ;
//		sort(copyed.begin(), copyed.end());
//		int len = nums.size()-1;
//		for (int i = 1; i < nums.size();i=i+2)
//	           //�����Ŵ��� 
//				nums[i] = copyed[len--];
//		
//		for (int i = 0; i < nums.size(); i=i+2)
//			 //ż����С�� 
//			 nums[i] = copyed[len--];
//	}
//
//};
//

//class Solution {
//public:
//	bool increasingTriplet(vector<int>& nums) {
//		if (nums.size() < 3) return false;
//		int firstmin = INT_MAX, secondmin = INT_MAX;
//		for (int i = 0; i < nums.size(); i++)
//		{
//			if (nums[i]<firstmin) 
//				firstmin = nums[i];
//			else 
//				if (nums[i] = firstmin) continue;
//				else 
//					if (nums[i] < secondmin) secondmin = nums[i];
//					else if (nums[i] > secondmin) return true;
//			
//		
//		}
//		return false;
//	}
//};


//// ����������������������ܿ��ѧ���ˣ�����˴򿨣�����ǿ�磬���ǲ��������̲� haha~
// struct TreeNode {
//     int val;
//     TreeNode *left;
//     TreeNode *right;
//     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
// };
// //�����ض��������������ʵ���Ͼ��Ǳ����õ������δ�С�������еĽڵ�ֵ
//class Solution {
//public:
//	vector<int>result;
//	int  kthSmallest(TreeNode* root,int k) {
//		if (root == NULL) return 0;
//		if (root->left != NULL)  //����з������
//			kthSmallest(root->left,1);
//		//���û�У������Լ��ˣ����Է�������ֵ������ұ�
//		result.push_back(root->val);
//		if (root->right != NULL) kthSmallest(root->right,1);
//		return result[k-1];
//	}
//};




////��Ҫ������ݹ��˼·�����뵽��Ҫ���մ��˼·��Ҫ����ϸ֦ĩ�ڣ�Ҫ�����Ʋ������ˡ��öิϰ���Ρ�
//class Solution {
//public:
//	TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
//		if (root == p || root == q) return root;//�ҵ���ָ���ڵ�ֱ�ӷ���
//		if (root == NULL) return NULL; //��Ѱ�����û����Ѱ����˵��û�ҵ�������NULL
//		//û���ҵ���Ҫ��ʼ��Ѱ��,�������
//		TreeNode *left = lowestCommonAncestor(root->left, p, q);
//		TreeNode *right = lowestCommonAncestor(root->right, p, q);
//
//		//����ԭ��
//		if (left == NULL&&right ==NULL) //��û������ 
//			return NULL;
//		if (left != NULL&&right == NULL)  //������û��
//			return left;
//		if (left == NULL&&right != NULL)
//			return right;
//		else return root; //���ж���
//	}
//};
//
//struct TreeNode {
//	int val;
//	TreeNode *left;
//	TreeNode *right;
//	TreeNode(int x) : val(x), left(NULL), right(NULL) {}
//
//};
//
//class Solution {
//public:
//	TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
//		if (root->val == p->val || root->val == q->val) return root;
//		else
//			if (root->val > p->val&&root->val > q->val) //��С�ڵ�ǰ�ڵ㣬�϶����ұ�
//			{
//				return lowestCommonAncestor(root->left, p, q);
//			}
//			else if (root->val < p->val&&root->val < q->val)
//			{
//				return lowestCommonAncestor(root->right, p, q);
//			}
//			else
//				return root;
//	}
//};


//��̬�滮�������������ΪС�����⡣dp[i]��ʾ������iλΪ�����е�����������������
//������һ״̬���� dp[i+1] Ҫ���£�nums[i+1]��Ҫ���κ�nums[i],nums[i-1]...nums[0]�Ƚ�������dp[i+1]
//����̬�滮��ʱ������Ҫ���߼����������д���롣
//class Solution {
//public:
//	int lengthOfLIS(vector<int>& nums) {
//		int rst = 1; int median = 1;
//		if (nums.size() == 0) return 0;
//		vector<int> dp(nums.size(), 1);
//		dp[0] = 1;
//		for (int i = 1; i < nums.size();i++)
//		{
//			for (int j = i - 1; j >= 0;j--)
//			{
//				//����i-2��0,�ҵ�dp[i]
//				if (nums[i] < nums[j]) median = 1;
//				else 
//					if (nums[i]>nums[j]) 
//				{
//					median = dp[j] + 1;
//				}
//				else
//				{   //�������
//					median = dp[j];
//				}
//				dp[i] = max(dp[i], median);
//			}
//			rst = max(rst, dp[i]);
//		}
//		return rst;
//
//	}
//};

// �����ⷨ�����������У�����ÿһ�������н����ж��Ƿ�����������������ж�size�Ƿ��󣬼�¼���µ�size
// ������һ��û��ͨ������ʱ��
//class Solution {
//public:
//	int longestSubstring(string s, int k) {
//		int rst = 0; int len = s.size();
//		if (s.size() == 0) return 0;
//		//map<int,int> times;
//		for (int i = 0; i < len; i++)
//		{
//			map<int, int> times;
//			times[s[i] - 'a']++; //��¼��ǰ�ַ�����
//			for (int j = i + 1; j < len; j++)
//			{
//				//������i����β���൱�ڱ��������е��Ӵ�
//				times[s[j] - 'a']++;
//				if (iscorrect(times, k) == true)  //��ǰ�Ӵ�������������¼��С;
//				{
//					rst = max(rst, j - i + 1);
//				}
//			}
//		}
//		if (k == 1)
//			return rst > 1 ? rst : 1;
//		else
//			return rst;
//	}
//	bool inline iscorrect(map<int, int> times, int k)
//	{
//		int lenth = times.size();
//		map<int, int>::iterator it = times.begin();
//		while (it != times.end())
//		{
//			if (it->second < k) return false;
//			it++;
//		}
//		return true;
//	}
//};

//����ⷨʵ����Ҳ�������汩�����Ż�����д���ж��Ƿ����������Ƕ�map���з��ʣ��ж�ÿ��Ԫ�صĳ��ִ���
//����ⷨ����һ�����������Ż�������һ��int����Ĥ����ÿһλ����1��0�Ĳ���
//��ĤΪ0��ʱ��Ҳ��������������ʱ�򣬾ͱ����˱���
//�ڶ��������ѭ����ֻҪi+k>n���Ͳ����˱����ˣ����ǵڶ����Ż�
//�������Ż�����i = i+max_idx,��������˰���ҲûŪ���ס�
// ��Ϊ�ҿ��˺ܾû�û��ȫ���ף�����ⲻ���Լ�д�ˡ�
//class Solution {
//public:
//	int longestSubstring(string s, int k) {
//		int res = 0, i = 0, n = s.size();
//		while (i + k <= n) {//�ؼ�
//			int m[26] = { 0 }, mask = 0, max_idx = i;
//			for (int j = i; j < n; ++j) {
//				int t = s[j] - 'a';
//				++m[t];
//				if (m[t] < k) mask |= (1 << t);//�ؼ�
//				else mask &= (~(1 << t));  //�ؼ�
//				if (mask == 0) {
//					res = max(res, j - i + 1);
//					max_idx = j;
//				}
//			}
//			i = i + max_idx; //û����������i++Ҳ��ͨ��
//		}
//		return res;
//	}
//};

//�Լ����д�ģ��������÷�����ѧϰһ�¾Ͳ�д��
//class Solution {
//public:
//	int getSum(int a, int b) {
//		if (b == 0) return a;
//		else
//			if (b>0)
//			{
//				while ((b--)!=0)
//				{
//					a++;
//				}
//				return a;
//			}
//			else
//			{
//				if (b < 0)
//				{
//					while ((b++) != 0)
//					{
//						a--;
//					}
//					return a;
//				}
//			}
//		return a;
//	}
////};
//
//class Solution {
//public:
//	int majorityElement(vector<int>& nums) {
//		int counttimes = 1;
//		if (nums.size() == 1) return nums[0];
//		int majority = nums[0];
//		for (int i = 1; i < nums.size();i++)
//		{
//			if (nums[i]==majority)
//				counttimes++;
//			else
//			{
//				if (counttimes ==0)
//				{
//					majority = nums[i];
//					counttimes++;
//				}
//				else
//				{//��Ϊmajority��countimesҲ��Ϊ0
//					counttimes--;
//				}
//			}
//			
//		}
//		return majority;
//	}
//};

//������ң����Լ򻯾Ͳ����ˣ�������Ҫ��������Ħ��ͶƱ����˼·
//class Solution {
//public:
//	vector<int> majorityElement(vector<int>& nums) {
//		int countx = 0, county = 0;
//		int majorityx = 0, majorityy = 0; vector<int>rst;
//		if (nums.size() == 0) return rst;
//		if (nums.size()==1)  
//		{
//			rst.push_back(nums[0]);
//			return rst;
//		}
//		else if (nums.size()==2)
//		{
//			if (nums[1] != nums[0])
//			{
//				rst.push_back(nums[0]);
//				rst.push_back(nums[1]);
//			}
//			else
//				rst.push_back(nums[0]);
//			return rst;
//		}
//		for (int i = 0; i < nums.size();i++)
//		{
//			
//			if (countx==0&&nums[i]!=majorityy)
//			{
//				majorityx = nums[i];
//				countx++;
//			}
//			else
//				if (nums[i] == majorityx)
//				{
//					countx++;
//				}
//				else//����Ϊ0ʱ
//					if (county == 0)
//					{
//						majorityy = nums[i];
//						county++;
//					}
//					else if (nums[i] == majorityy)
//					{
//						county++;
//					}
//					else
//					{
//						countx--; county--;
//					}
//		}
//		countx = 0; county = 0;
//		for (int i = 0; i < nums.size();i++)
//		{
//			if (majorityx==nums[i])
//			{
//				countx++;
//			}
//			else if (majorityy == nums[i])
//			{
//				county++;
//			}
//
//		}
//		if (countx>nums.size()/3)
//		{
//			rst.push_back(majorityx);
//		}
//		if (county > nums.size() / 3)
//		{
//			rst.push_back(majorityy);
//		}
//		return rst;
//	}
//};

//int calculate(string s) {
//	int res = 0, d = 0;
//	char sign = '+';
//	stack<int> nums;
//	for (int i = 0; i < s.size(); ++i) {
//		if (s[i] >= '0') {//ȡ����������
//			d = d * 10 - '0' + s[i];//��λ(�ȼ���),ȡ��һλ���ֶ����ֻ�ԭ
//		}
//		if ((s[i] < '0' && s[i] != ' ') || i == s.size() - 1) {  //�����ǰ�Ƿ���λ�������һλ�ˣ��ͻ��ж���һ�εķ���λsign
//			if (sign == '+') {
//				nums.push(d);
//			}
//			else if (sign == '-') {
//				nums.push(-d);
//			}
//			else if (sign == '*' || sign == '/') {
//				int tmp = sign == '*' ? nums.top() * d : nums.top() / d;
//				nums.pop();  //ȡ�������һ����������stack����ȳ�������
//				nums.push(tmp);
//			}
//			sign = s[i]; //���浱ǰ����
//			d = 0;
//		}
//	}
//	for (; !nums.empty(); nums.pop()) {
//		res += nums.top();
//	}
//	return res;
//}


//��ջ����Ŀ����Ϊ������˳��������Ǽӷ��Ļ���push��stack�о�û���Ⱥ�˳�򣬹ؼ�������*/
//���ǵ����ȼ���Ҫ��һЩ�����Ե�������*/ʱҪ��֮ǰpush��stack�е�����ȡ�������Ǻ�Ž�ȥ����ȡ
//����Ҫ��һ��stack��ȥ��������������֡� �������ó˷����������ʱ�򣬾Ͱ�stack�е�����pop����
//�������Ժ󽫳˷������Ľ��push��stack�У����stackԪ����Ӽ��ɡ�
//��һЩС��ϸ�ڱ������currentnumʱҪ��-��0����+s[i],��������Щ���ֻ�Խ�磡
//class Solution {
//public:
//	int calculate(string s) {
//		int len = s.size();
//		stack<int> nums;
//		int currentnum = 0;
//		int  rst = 0;
//		char presign = '+'; //����Ҫ����ʱ��һ�εķ���
//		for (int i = 0; i < len;i++)
//		{
//			if (s[i] >= '0'&&s[i] <= '9') //ȡ����Ԫ��������
//				currentnum = currentnum * 10 - '0' + s[i];//�ȼӷ��Ļ�����磡�ɿӡ�
//			
//			if (s[i] < '0'&&s[i] != ' '||i==len-1)  //ȡ���Ӽ��˳���ʾ��ǰ������ȡ���ˣ�����push��stack
//			{
//				if (presign == '+')
//				{
//					presign = s[i];
//					nums.push(currentnum);
//					currentnum = 0;
//				}
//				else if (presign == '-')
//				{
//					presign = s[i];
//					nums.push(-currentnum);
//					currentnum = 0;
//				}
//				else if (presign == '*')
//				{
//					presign = s[i];
//					currentnum = nums.top()*currentnum;
//					nums.pop();
//					nums.push(currentnum);
//					currentnum = 0;
//				}
//				else if (presign == '/')
//				{
//					presign = s[i];
//					currentnum = nums.top()/currentnum;
//					nums.pop();
//					nums.push(currentnum);
//					currentnum = 0;
//				}
//					
//			}
//		}
//
//		while (!nums.empty())
//		{
//			rst += nums.top();
//			nums.pop();
//		}
//		return rst;
//	}
//};

//ÿ�ο��Ž������׼����λ�þʹ��������������е�k�������
//���ǿ�����Ӵ�С���ţ����ÿ������������Ժ󷵻ص�Ԫ���ŵ�Ԫ������Ǵ������� �ұ���С�����ġ�
//����������±���index�Ļ� �������ǵ�index��1������� ÿ������ȷ����һ����INDEX�������
//Ȼ���k�Ƚϣ�����ȷ���µ���ʵ��������л����ұ����У��ټ������žͺ��ˡ�
//class Solution {
//public:
//	int findKthLargest(vector<int>& nums, int k) {
//		int low = 0, high = nums.size() - 1, mid = 0;
//		while (low <= high)
//		{
//			mid = partation(nums, low, high);  //��mid+1�����Ϊnums[mid]
//			if (mid + 1 == k) return nums[mid];
//			else if (mid + 1 < k)  //���ұ�
//			{
//				low = mid + 1; 
//			}
//			else
//			{
//				high = mid - 1;
//			}
//
//		}
//		return 1;
//	}
//
//	int partation(vector<int> &nums, int low, int high) //low��high������Ѱ�ķ�Χ
//	{
//		int left = low + 1, right = high;
//		int bound = nums[low];//�ο��ȶ���: bound
//		while (left <= right)
//		{
//			while (left < high && nums[left] > bound) left++;  //��ָ�����ң��ҵ���һ����boundС������ͣ��
//			while (nums[right] < bound &&right>low) right--;
//			if (left < right) swap(nums[left++], nums[right--]);
//			else break;
//		}
//		if (right < left)
//		{
//			swap(nums[low], nums[right]); return right;
//		}
//		else
//		{
//			swap(nums[low], nums[left]);
//			return left;  //���ص���Ҳ���ǵ�ǰ�ȶԲ��������꣬���������left+1�����
//		}
//	}
//};


// Employee info
//class Employee {
//public:
//// It's the unique ID of each node.
//// unique id of this employee
//int id;
//// the importance value of this employee
//int importance;
//// the id of direct subordinates
//vector<int> subordinates;
//Employee(int id, int importance, vector<int> subordinates)
//{
//	this->id = id;
//	this->importance = importance;
//	this->subordinates = subordinates;
//}
//};
//
//class Solution {
//public:
//	int getImportance(vector<Employee*> employees, int id) {
//		map<int, Employee*> employeemap ;//����һ��Ա��id����Ӧ���������ݽṹ��mapӳ���ϵ
//		map <int, bool> visited;  //����һ��visited
//		queue<int> BFSqueue;
//		for (int i = 0; i < employees.size(); i++)
//		{
//			employeemap[employees[i]->id] = employees[i];
//			visited[employees[i]->id] = false;
//		}
//
//		//BFS����,�Ӳ���������id��ʼ
//		BFSqueue.push(id); int importance=0;
//		while (!BFSqueue.empty())
//		{
//			int curid = BFSqueue.front();
//			visited[curid] = true; 
//			BFSqueue.pop(); importance += employeemap[curid]->importance;
//			for (int i = 0; i < (employeemap[curid]->subordinates).size(); i++)
//			{
//				//���ʵ��ǵ�ǰԱ���������ĵ�i���������
//				if (visited[(employeemap[curid]->subordinates)[i]] == false)
//				{
//					BFSqueue.push((employeemap[curid]->subordinates)[i]);
//				}
//			}
//		}
//		return importance;
//	}
//};
//class orange
//{
//public: 
//	int x;
//	int y;
//	int minute;
//	orange(int a, int b,int min):x(a), y(b),minute(min){}
//};
//class Solution {
//public:
//	int orangesRotting(vector<vector<int>>& grid) {
//		int row = grid.size(); int rst = 0;
//		int col = grid[0].size(); int counttotal = 0;
//		queue<orange> BFSqueue; int countrot = 0;
//		for (int i = 0; i < row; i++)
//		{
//			for (int j = 0; j < col; j++)
//			{
//				if (grid[i][j] == 2) //����Ǹ��õ����Ӿͼ��뵽����
//				{
//					BFSqueue.push(orange(i, j,0));  //��������Ϣ���뵽�����У��������
//					countrot++;
//				}
//				if (grid[i][j] == 1)
//				{
//					counttotal++;
//				}
//			}
//		}
//		counttotal += countrot;
//		
//		while (!BFSqueue.empty())
//		{
//			orange curcorange = BFSqueue.front();
//			int x = curcorange.x; int y = curcorange.y; int level = curcorange.minute;
//			BFSqueue.pop();
//			if (x+1 >= 0 && x + 1 <= row - 1 && y >= 0 && y <= col - 1 && grid[x + 1][y] == 1)
//			{
//				countrot++;
//				BFSqueue.push(orange(x + 1, y, level + 1));
//				rst = max(level + 1, rst);
//				grid[x + 1][y] = 2;
//			}
//			if (x - 1 >= 0 && x - 1 <= row - 1 && y >= 0 && y <= col - 1 && grid[x - 1][y] == 1)
//			{
//				countrot++;
//				BFSqueue.push(orange(x - 1, y, level + 1));
//				rst = max(level + 1, rst);
//				grid[x - 1][y] = 2;
//			}
//			if (x >= 0 && x <= row - 1 && y + 1 >= 0 && y + 1 <= col - 1 && grid[x ][y + 1] == 1)
//			{
//				countrot++;
//				BFSqueue.push(orange(x , y + 1, level + 1));
//				rst = max(level + 1, rst);
//				grid[x][y + 1] = 2;
//			}
//			if (x >= 0 && x <= row - 1 && y - 1 >= 0 && y - 1 <= col - 1 && grid[x ][y - 1] == 1)
//			{
//				countrot++;
//				BFSqueue.push(orange(x , y - 1, level + 1));
//				rst = max(level + 1, rst);
//				grid[x][y - 1] = 2;
//			}
//
//		}
//		if (countrot < counttotal) return -1;
//		else
//			return rst;
//	} 
//};

//struct TreeNode {
//	int val;
//	TreeNode *left;
//	TreeNode *right;
//	TreeNode(int x) : val(x), left(NULL), right(NULL) {}
//	
//};
//
//class Solution {
//public:
//	int pax = 0; int pay = 0; int levelx = 0; int levely = 0;
//	bool findx = false;	bool findy = false;
//	bool isCousins(TreeNode* root, int x, int y) {
//		findxy(root, root, x, y, 0);
//		return (levelx == levely&&pax != pay) ? true : false;
//
//	}
//	void findxy(TreeNode* curnode, TreeNode* previousNode, int x, int y, int level)
//	{
//		if (curnode == NULL) return;
//		if (curnode->val ==x)
//		{
//			cout << "�ҵ���x level=" << level << endl;
//			levelx = level; pax = previousNode->val; findx = true;
//		}
//		if (curnode->val == y)
//		{
//			cout << "�ҵ���y level=" << level << endl;
//			levelx = level; pay = previousNode->val; findy = true;
//		}
//		if (curnode->left != NULL && !(findx == true && findy == true))
//		{
//			cout << "Ѱ��" << curnode->val + "����ڵ�  level=" << level << endl;
//			findxy(curnode->left, curnode, x, y,level+1);
//			
//		}
//		if (curnode->right != NULL && !(findx == true && findy == true))
//		{
//			cout << "Ѱ��" << curnode->val + "���ҽڵ�  level=" << level << endl;
//			findxy(curnode->right, curnode, x, y, level + 1);
//		}
//	}
//};
//
//class Solution {
//	vector<vector<int>> rst;
//public:
//	vector<vector<int>> pathSum(TreeNode* root, int sum) {
//		vector<int>curpath; 
//		aimedpath(root, sum, curpath, 0);
//		return rst;
//	}
//
//	void aimedpath(TreeNode*root,int sum,vector<int> curpath,int cursum)
//	{
//
//		if (root == NULL) return;
//		cursum += root->val;
//		if (cursum == sum&&root->left==NULL&&root->right==NULL) {
//			curpath.push_back(root->val);  //��ǰ�ڵ���ȥ�Ͳ��ÿ�����
//			rst.push_back(curpath);
//			return;
//		}
//			curpath.push_back(root->val);
//			aimedpath(root->left, sum, curpath, cursum);
//			aimedpath(root->right, sum, curpath, cursum);
//	}
//};

//struct TreeNode {
//	     int val;
//	     TreeNode *left;
//	     TreeNode *right;
//	     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
//	 };
//
////�������������������
//class Solution {
//public:
//	int current; int previous; int rst = true;
//	bool isValidBST(TreeNode* root) {
//		current = root->val;
//		middleBFS(root);
//		return rst;
//	}
//
//	bool middleBFS(TreeNode* root)
//	{
//		if (rst == false) return false;
//		if (root == NULL) return;
//		if (root->left != NULL) middleBFS(root->left);
//		previous = current;
//		current = root->val;
//		if (current < previous) rst = false;
//		if (root->right != NULL) middleBFS(root->right);
//		return true;
//	}
//};
//

//��ѧϰ̰���㷨���Ӽ򵥵Ŀ�ʼ��������̫���ڼ򵥣����㲻��̰���㷨Ҳ֪����Ⱑ
//class Solution {
//public:
//	bool lemonadeChange(vector<int>& bills) {
//		if (bills[0] != 5 || bills[1] == 20)//�Լ���ʼΪ0�����Ե�һ�غϺ��Լ����е�ǮӦ��Ϊ5���ڶ��غ�ӦΪ10
//			return false;
//		int n = 0;//5Ԫ����Ŀ
//		int m = 0;//10Ԫ����Ŀ
//		for (int i = 0; i < bills.size(); i++)
//		{
//			if (bills[i] == 5)
//			{
//				n++;
//			}
//			else if (bills[i] == 10)
//			{
//				if (n)
//				{
//					n--;
//					m++;
//				}
//				else return false;
//			}
//			else
//			{
//				if (m)
//				{
//					if (n)
//					{
//						m--;
//						n--;
//						continue;
//					}
//				}
//				if (n >= 3)
//				{
//					n -= 3;
//				}
//				else return false;
//			}
//		}
//		return true;
//	}
//};

//Ҳ��̰���㷨��һ���Ӿ�д�����ˣ��о�̰���㷨�������һ��˼·��
//class Solution {
//public:
//	bool canJump(vector<int>& nums) {
//		if (nums.size() == 1) return true;;
//		int len = nums.size();
//		int curbest = len - 1;
//		for (int i = len-2; i>=1; i--)
//		{
//			if (nums[i] + i >= curbest)
//			{//��ǰλ�þ�����������λ�ã�����
//				curbest = i;
//			}
//		}
//		return nums[0] >= curbest ? true:false;
//	}
//};
//
//

//�����㷨,��������Ż����Ǽ�¼ÿһ�����λ�ã���¼������ǿ���ͨ���յ�Ļ�����˼·���´α�����ʱ��Ͳ����ߵ�ͷ��
//class Solution {
//public:
//	 //0 δ֪��1 
//	int jump(vector<int>& nums) {
//		int step = 0; int maxpos = 0; int len = nums.size() - 1;
//		for (int i = 0; i < len;)
//		{
//			int best = 0; int ini = 0;
//			
//			for (int j = 1; j <= nums[i];j++)  //�ڵ�i��ʱ��ǰ�ߵĲ���
//			{
//				if (i + j == len) return step + 1;
//
//				if (j+nums[i+j]>ini)  //��j����һЩ
//				{
//					best = j;  //����Ϊ��ǰ��j��
//					ini = j + nums[i + j];
//				}
//			}
//			i = i + best;
//			step++;
//		}
//		return step;
//	}
//
//	
//};


//ֻ������һ��� �������ű��˵�д��һ�顣����
//class Solution {
//public:
//	int videoStitching(vector<vector<int>>& clips, int T) {
//		if (T == 0)
//			return 0;
//		int len = clips.size();
//		int l, r;
//		l = 0;
//		r = T;
//		int re = 0;
//		while (l<r)
//		{
//			int maxl, maxr;
//			int indl, indr;
//			maxl = maxr = -1;
//			indl = l;
//			indr = r;
//			for (int i = 0; i<len; ++i)
//			{
//				if (clips[i][0] <= l)
//				{
//					if ((clips[i][1] - l)>maxl)
//					{
//						maxl = (clips[i][1] - l);
//						indl = i;
//					}
//				}
//				if (clips[i][1] >= r)
//				{
//					if ((r - clips[i][0])>maxr)
//					{
//						maxr = (r - clips[i][0]);
//						indr = i;
//					}
//				}
//			}
//			if (maxl == -1 || maxr == -1)
//				return -1;
//			++re;
//			if ((r) <= (clips[indl][1]) || l >= clips[indr][0])
//				break;
//			++re;
//			l = clips[indl][1];
//			r = clips[indr][0];
//		}
//		return re;
//	}
//};
//
//class Solution {
//public:
//	int Max1 = INT_MIN;
//	int maxPathSum(TreeNode* root) {
//		if (root == NULL) return INT_MIN;
//
//		digui(root);
//
//
//		return Max1;
//	}
//
//	int digui(TreeNode* root){
//		if (root == NULL) return INT_MIN;
//
//		int Max2 = root->val;
//		int Max3 = root->val;
//		int path1 = digui(root->left);
//		int path2 = digui(root->right);
//
//		if (path1 > 0)  //path1��ʾ���·��
//			Max2 = max(Max2, root->val + path1);
//		if (path2 > 0)  //�ұ�·��
//			Max2 = max(Max2, root->val + path2);
//		if (path1 > 0 && path2 > 0)  //
//			Max3 = root->val + path1 + path2;
//		Max1 = max(Max1, Max2);
//		Max1 = max(Max1, Max3);
//
//		cout << "�ڵ�" << root->val << "������" << Max1 << endl;
//		return Max2;
//	}
//
//};

//
//class Solution {
//public:
//	bool isPossible(vector<int>& nums) {
//		map <int, int> countnum;	map <int, int> tailmap;
//		for (int i = 0; i <= nums.size()-1;i++)
//		{
//			countnum[nums[i]]++;
//			
//		}
//	//����ÿһ�����У�ֻ�ü�ס��β�����о��У���Ϊ�������
//		for (int i = 0; i <= nums.size()-1;i++)
//		{
//			
//			if (countnum[nums[i]]==0)  continue;//û��Ԫ��
//			else
//			{
//				//�������Ԫ��
//				if(tailmap[nums[i]-1] == 0) //��βû��
//				{
//					if (countnum[nums[i] + 1] != 0 && countnum[nums[i] + 2] != 0)
//					{
//						//��������Ԫ�ض���
//						tailmap[nums[i] + 2]++; countnum[nums[i]]--;
//						countnum[nums[i] + 1]--; countnum[nums[i] + 2]--;
//					}
//					else return false;
//				}
//				else
//				{
//					//��β�����Ԫ�أ���ӵ���β����
//					tailmap[nums[i]]++; tailmap[nums[i]-1]--;
//					countnum[nums[i] ]--;
//				}
//
//			}
//		}
//		return true;
//	}
//};

//class Solution {
//public:
//	int candy(vector<int>& ratings) {
//		if (ratings.size() <= 1) return ratings.size();
//		int candies = 0; int up = 0; int old_slope = 0; int down = 0;
//		for (int i = 1; i <= ratings.size() - 1;i++)
//		{
//			//���ratings[i]>ratings[i-1],newslopΪ1������Ϊ0��С��Ϊ-1;
//			int new_slope = (ratings[i] > ratings[i - 1]) ? 1 : (ratings[i] < ratings[i - 1] ? -1 : 0);
//			if ((old_slope>0&&new_slope==0)||(old_slope<0&& new_slope>=0))
//			{
//				candies += countc(up) + countc(down) + max(up, down); up = 0; down = 0;
//			}
//			if (new_slope > 0) up++;
//			if (new_slope < 0) down++;
//			if (new_slope == 0) candies++;
//			old_slope = new_slope;
//		
//		}
//		candies = countc(up) + countc(down) + max(up, down) + 1;
//		return candies;
//	}
//
//	int countc(int n)
//	{
//		return n*(n + 1) / 2;
//	}
//};

// struct TreeNode {
//	int val;
//	TreeNode *left;
//	TreeNode *right;
//	TreeNode(int x) : val(x), left(NULL), right(NULL) {}
//	
//};
// class Solution {
// public:
//	 TreeNode* constructMaximumBinaryTree(vector<int>& nums) {
//		 TreeNode *root = new TreeNode(0);
//		 return construct(0, nums.size() - 1, nums);
//	 }
//
//	 TreeNode* construct(int le, int ri, vector<int>& nums)
//	 {
//		 if (le<0 || ri>nums.size() - 1 || le > ri) { cout << "le:" << le << " ri:" << ri << "����" << endl; return NULL; }
//		 if (le == ri) { TreeNode *node = new TreeNode(nums[le]); return node; }
//		 int maxnum = INT_MIN; int index = le;
//		 //��һ�������ǲ���ĵ�ǰ�ڵ㣬�ڶ��������ǿ�ʼλ�ã������������ǽ���λ��
//		 for (int i = le; i <= ri; i++)
//		 {
//			 if (nums[i] > maxnum)
//			 {
//				 index = i; maxnum = nums[i];
//			 }
//		 }
//		 cout << "le:" << le << " ri:" << ri << "Index:" << index << "������" << nums[index] << endl;
//		 TreeNode *node = new TreeNode(nums[index]);
//		 node->left = construct(le, index - 1, nums);
//		 node->right = construct(index + 1, ri, nums);
//		 return node;
//	 }
// };
//
//class Solution {
//public:
//	int uniquePaths(int m, int n) {
//		int left = m - 1, total = m + n - 2;
//		return zuhe(total, left);
//	}
//	int zuhe(int n,int m)
//	{
//		int rst = 1;
//		for (int i = m + 1; i <=n ; i++)
//		{
//			rst *= i;
//		}
//		for (int i = 2; i <= n-m; i++)
//		{
//			rst /= i;
//		}
//		return rst;
//	}
//};

//class Solution {
//public:
//	vector<string> rst;
//	vector<string> letterCombinations(string digits) {
//		vector<string> indexstring = { "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz" };
//		if (digits == "") return rst;
//	
//		digui(digits, 0, "",indexstring);
//		return rst;
//	
//	}
//	void digui(string digits, int index,string custring,vector<string> indexstring )
//	{
//		if (index == digits.size())
//		{
//			rst.push_back(custring); return;
//		}
//		for (int i = 0; i < indexstring[digits[index]-'2'].size();i++)
//		{
//			string custring1;
//			custring1=custring + indexstring[digits[index] - '2'][i];
//			digui(digits, index + 1, custring1, indexstring);
//			
//		}
//	}
//};


//class Solution {
//public:
//	vector<int> spiralOrder(vector<vector<int>>& matrix) {
//		vector<int>rst;
//		int right  = matrix[0].size() - 1;
//		int down = matrix.size() - 1;
//		int left = 0;
//		int up = 0;
//		while (1)
//		{
//			//���ң���matrix[left]�ߵ�
//			for (int i = left; i <= right; i ++ ) rst.push_back(matrix[up][i]);
//			if (++up > down) return rst;
//			for (int i = up; i <= down; i++) rst.push_back(matrix[i][right]);
//			if (--right<left) return rst;
//			for (int i = right; i >= left; i--) rst.push_back(matrix[down][i]);
//			if (--down<up)return rst;
//			for (int i = down; i >=up; i--) rst.push_back(matrix[i][left]);
//			if (++left > right) return rst;
//		}
//		return rst;
//	}
////};
//class Solution {
//public:
//	vector<vector<int>> generateMatrix(int n) {
//		vector<vector<int>> rst(n, vector<int>(n)); int cur = 1;
//		if (n == 1){ rst[0][0] = 1; return rst; }
//		int left = 0; int right = n - 1; int down = n-1; int up = 0;
//		while (1)
//		{
//			for (int i = left; i <= right; i++) rst[up][i] = cur++;
//			up++; if (cur > n*n) break;
//			for (int i = up; i <= down; i++) rst[i][right] = cur++;
//			right--; if (cur > n*n) break;
//			for (int i = right; i >= left; i--) rst[down][i] = cur++;
//			down--; if (cur > n*n) break;
//			for (int i = down; i >= up; i--)
//				rst[i][left] = cur++;
//			left++; if (cur > n*n) break;
//		}
//		return rst;
//	}
//};
//
//class Solution {
//public:
//	int firstUniqChar(string s) {
//		map<char, int> coutt; int curindex = -1;
//		for (int i = 0; i < s.size(); i++)
//		{
//			coutt[s[i]]++;
//		}
//		for (int i = 0; i < s.size(); i++)
//		{
//			if (coutt[s[i]] == 1) return i;
//		}
//		return -1;
//	}
//};

//
//class Solution {
//public:
//	int evalRPN(vector<string>& tokens) {
//		stack<int> s; int rst = 0;
//		if (tokens.size() == 1) return atoi(tokens[0].c_str());
//		for (int i = 0; i < tokens.size();i++)
//		{
//			string str = tokens[i];
//			if (str == "+")
//			{
//				int b = s.top(); s.pop();
//				int a = s.top(); s.pop();
//				rst = a + b;
//				s.push(rst); continue;
//			}
//			if (str == "-")
//			{
//				int b = s.top(); s.pop();
//				int a = s.top(); s.pop();
//				rst = a - b;
//				s.push(rst); continue;
//			}
//			if (str == "*")
//			{
//				int b = s.top(); s.pop();
//				int a = s.top(); s.pop();
//				rst = a*b;
//				s.push(rst); continue;
//			}
//			if (str == "/")
//			{
//				int b = s.top(); s.pop();
//				int a = s.top(); s.pop();
//				rst = a / b;
//				s.push(rst); continue;
//			}
//			s.push(atoi(str.c_str()));
//		}
//		return rst;
//	}
//};

//class Solution {
//public:
//	bool flag = false;
//	bool wordBreak(string s, vector<string>& wordDict) {
//		int len = s.size(); 
//		if (s.size() == 0 || wordDict.size() == 0) return false;
//			
//		digui(s, 0, len, wordDict);
//		return flag;
//	}
//	void digui(string s, int index, int len, vector<string> wordDict)
//	{
//		if (index == len) 
//			flag = true;
//		if (flag == true) return;
//		for (int i = 0; i < wordDict.size(); i++)
//		{
//			string curstring = wordDict[i];
//			if (index+curstring.size()>len) continue;
//			else
//				if (s.substr(index, curstring.size()) == curstring)
//				{
//					digui(s, index + curstring.size(), len, wordDict);
//				}
//		}
//	}
//};


//class Solution {
//public:
//	bool flag = false;
//	
//	bool wordBreak(string s, vector<string>& wordDict) {
//		int len = s.size();
//		if (s.size() == 0 || wordDict.size() == 0) return false;
//		vector<int> rem(s.size(), -1);
//		digui(s, 0, len, wordDict,rem);
//		return flag;
//	}
//	void digui(string s, int index, int len, vector<string> wordDict,vector<int> &rem)
//	{
//		if (index == len)
//			flag = true;
//		if (flag == true||rem[index]==0) return ;
//		for (int i = 0; i < wordDict.size(); i++)
//		{
//			string curstring = wordDict[i];
//			if (index + curstring.size()>len) continue;
//			else
//				if (s.substr(index, curstring.size()) == curstring)
//				{
//					digui(s, index + curstring.size(), len, wordDict, rem);
//				}
//		}
//		if (flag == false) rem[index] = 0;
//	}
//};

//class Solution {
//public:
//	vector<string> wordBreak(string s, vector<string>& wordDict) {
//		int len = s.size();
//		vector<string> rst; if (s.size() == 0 || wordDict.size() == 0) return rst;
//		vector<int> rem(s.size(), -1); bool thistime = false;
//		digui(s, 0, len, wordDict, rem, "", rst, thistime);
//		return rst;
//	
//	}
//	void digui(string s, int index, int len, vector<string> wordDict, vector<int> &rem,string curs,vector<string>& rst,bool &thistime)
//	{
//		
//		if (index == len) { rst.push_back(curs); thistime = true;  return; }
//		if (rem[index] == 0) 
//			return;
//		for (int i = 0; i < wordDict.size(); i++)
//		{
//			string curs1 = curs;
//			string curstring = wordDict[i];
//			if (index + curstring.size()>len) continue;
//			else
//			{
//				if (s.substr(index, curstring.size()) == curstring)
//				{
//					if (index == 0) curs1 = curstring;
//					else
//						curs1 += " " + curstring;
//					digui(s, index + curstring.size(), len, wordDict, rem, curs1,rst,thistime);
//				}
//			}
//		}
//		if (thistime == false)
//			rem[index] = 0;
//		
//		
//	} 
//};

class Solution {
public:
	vector<vector<string>> rst;
	vector<vector<string>> partition(string s) {
		vector<string> cur;
		digui(0, cur, s);
		return rst;
	}

	void digui(int index, vector<string>cur,string s)
	{
		if (index >= s.size()) {
			rst.push_back(cur); return;
		}
		for (int i = 1; (int)(s.size())-i-index>=0;i++)
		{
			string sub = s.substr(index, i);
			if (isHui(sub) == true) {
				cur.push_back(sub);
				digui(index + i, cur, s);
				cur.pop_back();
			}
		}
	}

	bool isHui(string s)
	{
		int first = 0; int last = s.size()-1;
		while (first<last)
		{
			if (s[first] == s[last]) { first++; last--; continue; }
			else
				return false;
		}
		return true;
	}
};

int main(int argc, char *argv[])
{
	string s = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
	vector<string> wordDict = { "a", "aa", "aaa", "aaaa", "aaaaa" };
	Solution t;
	t.partition("efe");
}
	















