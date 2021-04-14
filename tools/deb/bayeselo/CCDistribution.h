/////////////////////////////////////////////////////////////////////////////
//
// Rémi Coulom
//
// December, 2004
//
/////////////////////////////////////////////////////////////////////////////
#ifndef CCDistribution_Declared
#define CCDistribution_Declared

#include "CDistribution.h"
#include "CDiscretization.h"

class CCDistribution: public CDistribution, public CDiscretization // cdist
{
 private: ///////////////////////////////////////////////////////////////////
  double GetBoundValue(double Confidence,
                       int Begin,
                       int End,
                       int Direction) const;

 public: ////////////////////////////////////////////////////////////////////
  CCDistribution(int Size, double Min, double Max):
   CDistribution(Size),
   CDiscretization(Size, Min, Max)
  {}

  //
  // Compute properties of this distribution
  //
  double GetLowerValue(double Confidence) const;
  double GetUpperValue(double Confidence) const;
  double GetMostLikelyValue() const;
  double GetMean() const;
  double GetVariance(double Mean) const;

  //
  // Standard distributions
  //
  void SetNormal(double Mu, double Sigma);
  void SetBinomialPosterior(int N, int n);
};

#endif
