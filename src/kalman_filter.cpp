#include "kalman_filter.h"

#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

/*
 * Please note that the Eigen library does not initialize
 *   VectorXd or MatrixXd objects with zeros upon creation.
 */

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict()
{
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {

  /* For lidar measurements, the error equation is y = z - H * x'. */
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;


  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

// Actually represents h(x')
VectorXd KalmanFilter::h(const VectorXd& state)
{
   const double px{state(0)};
   const double py{state(1)};
   const double vx{state(2)};
   const double vy{state(3)};

   double rho = sqrt(px * px + py * py);
   const double phi = std::atan2(py, px);

   /*
      Avoid Divide by Zero throughout the Implementation

      Before and while calculating the Jacobian matrix Hj, make sure your code avoids dividing by zero.
      For example, both the x and y values might be zero or px*px + py*py might be close to zero.
      What should be done in those cases?
   */
   if (rho <= 0)
   {
      rho = 1;
   }

   const double rho_dot = (px * vx + py * vy) / rho;

   VectorXd polar_state = VectorXd(3);
   polar_state << rho, phi, rho_dot;

   return polar_state;
}

/*
HINT: When working in radians, you can add 2π2\pi2π or subtract 2π2\pi2π until the angle is within the desired range.
*/
void KalmanFilter::NormalizeAngle(double& angle)
{
   while (angle > M_PI || angle < -M_PI)
   {
      if (angle > M_PI)
      {
         angle -= 2 * M_PI;
      }
      else
      {
         angle += 2 * M_PI;
      }
   }
}

void KalmanFilter::UpdateEKF(const VectorXd &z)
{
   /*

   For radar measurements, the functions that map the x vector [px, py, vx, vy] to polar coordinates are non-linear.
   Instead of using H to calculate y = z - H * x',
   for radar measurements you'll have to use the equations that map from cartesian to polar coordinates: y = z - h(x').

   */
  VectorXd x_f = h(x_);

  VectorXd y = z - x_f;

  /*
   In C++, atan2() returns values between -pi and pi.
   When calculating phi in y = z - h(x) for radar measurements,
   the resulting angle phi in the y vector should be adjusted so that it is between -pi and pi.
   The Kalman filter is expecting small angle values between the range -pi and pi.
   */
  auto& phi = y(1);
  NormalizeAngle(phi);

  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}
