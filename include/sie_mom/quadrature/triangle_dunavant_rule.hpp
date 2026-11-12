int dunavant_degree ( int rule );
int dunavant_order_num ( int rule );
void dunavant_rule ( int rule, int order_num, double xy[], double w[] );
int dunavant_rule_num ( );
int *dunavant_suborder ( int rule, int suborder_num );
int dunavant_suborder_num ( int rule );
void dunavant_subrule ( int rule, int suborder_num, double suborder_xyz[], 
  double suborder_w[] );
void dunavant_subrule_01 ( int suborder_num, double suborder_xyz[], 
  double suborder_w[] );
void dunavant_subrule_02 ( int suborder_num, double suborder_xyz[], 
  double suborder_w[] );
void dunavant_subrule_03 ( int suborder_num, double suborder_xyz[], 
  double suborder_w[] );
void dunavant_subrule_04 ( int suborder_num, double suborder_xyz[], 
  double suborder_w[] );
void dunavant_subrule_05 ( int suborder_num, double suborder_xyz[], 
  double suborder_w[] );
void dunavant_subrule_06 ( int suborder_num, double suborder_xyz[], 
  double suborder_w[] );
void dunavant_subrule_07 ( int suborder_num, double suborder_xyz[], 
  double suborder_w[] );
void dunavant_subrule_08 ( int suborder_num, double suborder_xyz[], 
  double suborder_w[] );
void dunavant_subrule_09 ( int suborder_num, double suborder_xyz[], 
  double suborder_w[] );
void dunavant_subrule_10 ( int suborder_num, double suborder_xyz[], 
  double suborder_w[] );
void dunavant_subrule_11 ( int suborder_num, double suborder_xyz[], 
  double suborder_w[] );
void dunavant_subrule_12 ( int suborder_num, double suborder_xyz[], 
  double suborder_w[] );
void dunavant_subrule_13 ( int suborder_num, double suborder_xyz[], 
  double suborder_w[] );
void dunavant_subrule_14 ( int suborder_num, double suborder_xyz[], 
  double suborder_w[] );
void dunavant_subrule_15 ( int suborder_num, double suborder_xyz[], 
  double suborder_w[] );
void dunavant_subrule_16 ( int suborder_num, double suborder_xyz[], 
  double suborder_w[] );
void dunavant_subrule_17 ( int suborder_num, double suborder_xyz[], 
  double suborder_w[] );
void dunavant_subrule_18 ( int suborder_num, double suborder_xyz[], 
  double suborder_w[] );
void dunavant_subrule_19 ( int suborder_num, double suborder_xyz[], 
  double suborder_w[] );
void dunavant_subrule_20 ( int suborder_num, double suborder_xyz[], 
  double suborder_w[] );
int i4_max ( int i1, int i2 );
int i4_min ( int i1, int i2 );
int i4_modp ( int i, int j );
int i4_wrap ( int ival, int ilo, int ihi );
double r8_huge ( );
int r8_nint ( double x );
void reference_to_physical_t3 ( double t[], int n, double ref[], double phy[] );
int s_len_trim ( char *s );
void timestamp ( );
double triangle_area ( double t[2*3] );

#include <stdexcept>
#include <vector>

namespace sie_mom::quadrature {

/**
 * @brief Quadrature points and weights for Dunavant triangle rules.
 * points contains 2 * order values (barycentric coordinates on reference triangle).
 * weights contains order values summing to 1.
 */
struct QuadratureData {
    int order = 0;
    std::vector<float> points;
    std::vector<float> weights;
};

/**
 * @brief Generates Dunavant quadrature points and weights in float precision for GPU kernels.
 *
 * @param rule Dunavant rule index in [1, 20].
 * @return QuadratureData with reference points and weights.
 */
inline QuadratureData make_quadrature(int rule) {
    if (rule < 1 || rule > 20) {
        throw std::invalid_argument("Dunavant quadrature rule must be in [1, 20]");
    }
    QuadratureData quad;
    quad.order = dunavant_order_num(rule);

    std::vector<double> points_double(2 * quad.order);
    std::vector<double> weights_double(quad.order);
    dunavant_rule(rule, quad.order, points_double.data(), weights_double.data());

    quad.points.resize(2 * quad.order);
    quad.weights.resize(quad.order);
    for (int i = 0; i < quad.order; ++i) {
        quad.points[2 * i] = static_cast<float>(points_double[2 * i]);
        quad.points[2 * i + 1] = static_cast<float>(points_double[2 * i + 1]);
        quad.weights[i] = static_cast<float>(weights_double[i]);
    }
    return quad;
}

} // namespace sie_mom::quadrature
