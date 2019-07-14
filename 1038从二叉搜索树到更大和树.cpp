#include<vector>
#include <fstream>
#include<iostream>
#include<map>
#include<string.h>
using namespace std;
class Solution {
public:
	int sum =0;
    TreeNode* bstToGst(TreeNode* root) {
        if(root==NULL) return NULL;
        bstToGst(root->right);
        sum+=root->val;
        root->val =sum;
        bstToGst(root->left);
    }
};
int main(int argc, char* argv[])
{
    Solution t;
    cout<<t.uniquePaths(7,3);
}
