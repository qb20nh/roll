#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <time.h>
#include <omp.h>

/* * CONFIGURATION 
 * FAST_MODE 1: Only checks partitions containing at least one 5-digit number. (Instant)
 * FAST_MODE 0: Checks ALL 8.5 billion combinations. (Takes 30s-3mins depending on CPU)
 */
#define FAST_MODE 1

// --- CONSTANTS ---
#define NUM_DIGITS 9
#define NUM_OPERANDS 5
#define NUM_OPERATORS 4
#define RPN_LEN 9

// Total permutations of 9 digits: 362,880
#define NUM_DIGIT_PERMS 362880 
// Ways to split 9 digits into 5 groups: 70
#define NUM_SPLITS 70 
// Permutations of 4 operators: 24
#define NUM_OP_PERMS 24 
// Valid RPN shapes for 5 numbers: 14
#define NUM_RPN_SHAPES 14 

// --- GLOBALS & LOOKUP TABLES ---

// Stores all 362,880 permutations of digits 1-9
unsigned char digit_perms[NUM_DIGIT_PERMS][NUM_DIGITS];

// Stores the 70 ways to split digits. 
// Format: 4 indices indicating where to cut. e.g., {1, 2, 3, 4}
int split_indices[NUM_SPLITS][4];

// Stores the 24 permutations of operators. 
// 0=+, 1=-, 2=*, 3=/
int op_perms[NUM_OP_PERMS][NUM_OPERATORS];

// Stores the 14 valid RPN shapes.
// 0 = Operand, 1 = Operator
int rpn_shapes[NUM_RPN_SHAPES][RPN_LEN];

// Global Max Tracking
double global_max_val = -DBL_MAX;
char global_best_expr[256];

// --- HELPER FUNCTIONS ---

void swap(unsigned char *a, unsigned char *b) {
    unsigned char temp = *a;
    *a = *b;
    *b = temp;
}

// Generate permutations recursively (used for initialization)
void generate_digit_perms_recursive(unsigned char *arr, int l, int r, int *count) {
    if (l == r) {
        memcpy(digit_perms[*count], arr, NUM_DIGITS);
        (*count)++;
    } else {
        for (int i = l; i <= r; i++) {
            swap((arr + l), (arr + i));
            generate_digit_perms_recursive(arr, l + 1, r, count);
            swap((arr + l), (arr + i)); // backtrack
        }
    }
}

// Generate splits (combinations of cut points)
void generate_splits_recursive(int start, int depth, int *current, int *count) {
    if (depth == 4) {
        memcpy(split_indices[*count], current, 4 * sizeof(int));
        (*count)++;
        return;
    }
    for (int i = start; i <= 8; i++) {
        current[depth] = i;
        generate_splits_recursive(i + 1, depth + 1, current, count);
    }
}

// Generate Op Perms
void generate_op_perms_recursive(int *arr, int l, int r, int *count) {
    if (l == r) {
        memcpy(op_perms[*count], arr, NUM_OPERATORS * sizeof(int));
        (*count)++;
    } else {
        for (int i = l; i <= r; i++) {
            int temp = arr[l]; arr[l] = arr[i]; arr[i] = temp;
            generate_op_perms_recursive(arr, l + 1, r, count);
            temp = arr[l]; arr[l] = arr[i]; arr[i] = temp;
        }
    }
}

// RPN Generator
void generate_rpn_recursive(int *current, int len, int zeros, int ones, int *count) {
    if (len == 9) {
        if (zeros == 5 && ones == 4) {
            memcpy(rpn_shapes[*count], current, 9 * sizeof(int));
            (*count)++;
        }
        return;
    }
    // Add Operand (0)
    if (zeros < 5) {
        current[len] = 0;
        generate_rpn_recursive(current, len + 1, zeros + 1, ones, count);
    }
    // Add Operator (1) - Logic: Stack must have at least 2 items (zeros - ones >= 2)
    if ((zeros - ones) >= 2 && ones < 4) {
        current[len] = 1;
        generate_rpn_recursive(current, len + 1, zeros, ones + 1, count);
    }
}

void init_tables() {
    // 1. Digit Permutations
    unsigned char d[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    int count = 0;
    generate_digit_perms_recursive(d, 0, 8, &count);
    
    // 2. Splits
    int s_curr[4];
    count = 0;
    generate_splits_recursive(1, 0, s_curr, &count);

    // 3. Op Perms
    int ops[] = {0, 1, 2, 3}; // +, -, *, /
    count = 0;
    generate_op_perms_recursive(ops, 0, 3, &count);

    // 4. RPN Shapes
    int rpn_curr[9];
    count = 0;
    generate_rpn_recursive(rpn_curr, 0, 0, 0, &count);
}

// --- EVALUATION ---

// Build readable string (only called when a new max is found)
void build_expression_string(char *buffer, int rpn_idx, double *nums, int op_perm_idx) {
    char stack[9][64];
    int sp = 0;
    int num_idx = 0;
    int op_idx = 0;
    
    char op_syms[] = {'+', '-', '*', '/'};
    
    for (int i = 0; i < 9; i++) {
        if (rpn_shapes[rpn_idx][i] == 0) {
            // Push number
            sprintf(stack[sp++], "%.0f", nums[num_idx++]);
        } else {
            // Operator
            char *right = stack[--sp];
            char *left = stack[--sp];
            int op = op_perms[op_perm_idx][op_idx++];
            
            char temp_buf[64];
            sprintf(temp_buf, "(%s %c %s)", left, op_syms[op], right);
            strcpy(stack[sp], temp_buf);
            sp++;
        }
    }
    strcpy(buffer, stack[0]);
}

int main() {
    printf("Initializing lookup tables...\n");
    init_tables();
    
    printf("Tables ready. Digits: %d, Splits: %d, Ops: %d, RPNs: %d\n", 
           NUM_DIGIT_PERMS, NUM_SPLITS, NUM_OP_PERMS, NUM_RPN_SHAPES);
    
    printf("Starting Search (Mode: %s)...\n", FAST_MODE ? "FAST (Heuristic)" : "FULL (Exhaustive)");
    
    double start_time = omp_get_wtime();

    // Parallelize the outer loop (Digit Permutations)
    #pragma omp parallel
    {
        double local_max = -DBL_MAX;
        char local_expr[256];
        double numbers[5];
        
        int dp;
        #pragma omp for schedule(dynamic, 1000)
        for (dp = 0; dp < NUM_DIGIT_PERMS; dp++) {
            
            // For each split pattern
            for (int sp = 0; sp < NUM_SPLITS; sp++) {
                
                // Construct numbers from digits based on split indices
                // digit_perms[dp] is the array of 9 digits
                // split_indices[sp] are the 4 cut points (e.g., 1, 2, 3, 4)
                
                int current_digit = 0;
                int prev_cut = 0;
                int has_large_num = 0;

                for (int n = 0; n < 5; n++) {
                    int cut = (n < 4) ? split_indices[sp][n] : 9;
                    double val = 0;
                    
                    // Convert slice to number
                    for (int k = prev_cut; k < cut; k++) {
                        val = val * 10.0 + digit_perms[dp][k];
                    }
                    numbers[n] = val;
                    prev_cut = cut;
                    
                    if (FAST_MODE) {
                        if (val > 9999) has_large_num = 1;
                    }
                }

                // Optimization: Skip if no 5-digit number (if FAST_MODE enabled)
                if (FAST_MODE && !has_large_num) continue;

                // For each operator permutation
                for (int op = 0; op < NUM_OP_PERMS; op++) {
                    
                    // For each RPN shape
                    for (int rp = 0; rp < NUM_RPN_SHAPES; rp++) {
                        
                        // Evaluate RPN
                        double stack[5];
                        int sp_idx = 0;
                        int num_idx = 0;
                        int op_idx = 0;
                        int error = 0;
                        
                        for (int token = 0; token < 9; token++) {
                            if (rpn_shapes[rp][token] == 0) {
                                stack[sp_idx++] = numbers[num_idx++];
                            } else {
                                double r = stack[--sp_idx];
                                double l = stack[--sp_idx];
                                int operator_type = op_perms[op][op_idx++];
                                double res = 0;
                                
                                switch (operator_type) {
                                    case 0: res = l + r; break; // +
                                    case 1: res = l - r; break; // -
                                    case 2: res = l * r; break; // *
                                    case 3: // /
                                        if (r == 0) { error = 1; goto end_rpn; }
                                        res = l / r; 
                                        break;
                                }
                                stack[sp_idx++] = res;
                            }
                        }
                        
                        if (stack[0] > local_max) {
                            local_max = stack[0];
                            // Reconstruct string only when we find a new max
                            build_expression_string(local_expr, rp, numbers, op);
                        }
                        
                        end_rpn:;
                    }
                }
            }
        }
        
        // Merge results thread-safely
        #pragma omp critical
        {
            if (local_max > global_max_val) {
                global_max_val = local_max;
                strcpy(global_best_expr, local_expr);
            }
        }
    }

    double end_time = omp_get_wtime();
    
    printf("\n--- FINAL RESULT ---\n");
    printf("Max Value: %.0f\n", global_max_val); // Print as integer if whole
    printf("Expression: %s\n", global_best_expr);
    printf("Time Taken: %.4f seconds\n", end_time - start_time);

    return 0;
}