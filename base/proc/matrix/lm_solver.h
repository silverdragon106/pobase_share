#pragma once

#include "define.h"
#include "matrix.h"

class CLMSolver
{
public:
	enum { LM_CALC_J = 0, LM_CALC_ERR = 1, LM_CALC_FINISHED = 2, LM_CALC_FAILED = 3};

	CLMSolver(i32 param_count, i32 sample_count, i32 max_iteration, f64 err_threshold);
	~CLMSolver();
	
	void					lambdaInit(i32 min_lambda, i32 max_lambda);
	i32						getStatus();
	
	bool					update(f64* model_param_ptr, const PoMat& matJ, const PoMat& matErr, f64 err_norm);
	bool					step(f64* model_param_ptr);

public:
	i32						m_param_count;
	i32						m_max_iteration;
	f64						m_err_threshold;
	i32						m_status;
	i32						m_lambda_lg10;
	i32						m_min_lambda;
	i32						m_max_lambda;

	i32						m_iteration;
	f64						m_prev_err_norm;
	f64*					m_prev_param_ptr;

	PoMat					m_matJt;
	PoMat					m_matJtJ;
	PoMat					m_matJtErr;
	PoMat					m_matJtJDiag;
	PoMat					m_matJtJI;
	PoMat					m_matD;
};

class CGNSolver
{
public:
	enum { GN_CALC_JERR = 0, GN_CALC_FINISHED = 1, GN_CALC_FAILED = 3 };

	CGNSolver(i32 param_count, i32 sample_count, i32 max_iteration, f64 err_threshold);
	~CGNSolver();

	i32						getStatus();
	bool					update(f64* model_param_ptr, const PoMat& matJ, const PoMat& matErr, f64 err_norm);

public:
	i32						m_param_count;
	i32						m_max_iteration;
	f64						m_err_threshold;

	i32						m_status;
	i32						m_iteration;
	f64						m_prev_err_norm;
	f64*					m_prev_param_ptr;

	PoMat					m_matJt;
	PoMat					m_matJtJ;
	PoMat					m_matJtErr;
	PoMat					m_matJtJI;
	PoMat					m_matD;
};