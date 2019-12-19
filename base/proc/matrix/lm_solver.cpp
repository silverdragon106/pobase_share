#include "lm_solver.h"
#include "base.h"

CLMSolver::CLMSolver(i32 param_count, i32 sample_count, i32 max_iteration, f64 err_threshold)
{
	m_param_count = param_count;
	m_max_iteration = max_iteration;
	m_err_threshold = err_threshold;
	m_status = CLMSolver::LM_CALC_J;
	m_lambda_lg10 = -3;
	m_min_lambda = -16;
	m_max_lambda = 16;

	m_iteration = 0;
	m_prev_err_norm = DBL_MAX;
	m_prev_param_ptr = new f64[param_count];
	memset(m_prev_param_ptr, 0, sizeof(f64)*param_count);

	CPOMatrix::initMatrix(m_matJt,param_count, sample_count, PO_TYPE_F64);
	CPOMatrix::initMatrix(m_matJtJ, param_count, param_count, PO_TYPE_F64);
	CPOMatrix::initMatrix(m_matJtJDiag, param_count, param_count, PO_TYPE_F64);
	CPOMatrix::initMatrix(m_matJtJI, param_count, param_count, PO_TYPE_F64);
	CPOMatrix::initMatrix(m_matJtErr, param_count, 1, PO_TYPE_F64);
	CPOMatrix::initMatrix(m_matD, param_count, 1, PO_TYPE_F64);
}

CLMSolver::~CLMSolver()
{
	POSAFE_DELETE_ARRAY(m_prev_param_ptr);
	CPOMatrix::freeMatrix(m_matJt);
	CPOMatrix::freeMatrix(m_matJtJ);
	CPOMatrix::freeMatrix(m_matJtErr);
	CPOMatrix::freeMatrix(m_matJtJDiag);
	CPOMatrix::freeMatrix(m_matJtJI);
	CPOMatrix::freeMatrix(m_matD);
}

i32 CLMSolver::getStatus()
{
	return m_status;
}

bool CLMSolver::update(f64* model_param_ptr, const PoMat& matJ, const PoMat& matErr, f64 err_norm)
{
	if (m_status >= CLMSolver::LM_CALC_FINISHED || !model_param_ptr)
	{
		return false;
	}

	if (m_status == CLMSolver::LM_CALC_J)
	{
		CPOMatrix::transpose(matJ, m_matJt);
		CPOMatrix::multiply(m_matJt, matErr, m_matJtErr);
		CPOMatrix::multiply(m_matJt, matJ, m_matJtJ);

		m_prev_err_norm = err_norm;
		CPOBase::memCopy(m_prev_param_ptr, model_param_ptr, m_param_count);

		step(model_param_ptr);
		m_status = CLMSolver::LM_CALC_ERR;
		return true;
	}
	if (err_norm > m_prev_err_norm)
	{
		if (++m_lambda_lg10 <= m_max_lambda)
		{
			step(model_param_ptr);
			m_status = CLMSolver::LM_CALC_ERR;
			return true;
		}
	}

	//update lambda
	m_lambda_lg10 = po::_max(m_lambda_lg10 - 1, m_min_lambda);

	//check terminate condition
	if (++m_iteration > m_max_iteration || err_norm < m_err_threshold ||
		CPOMatrix::norm(m_prev_param_ptr, model_param_ptr, m_param_count, PO_NORM_L2) < m_err_threshold)
	{
		m_status = CLMSolver::LM_CALC_FINISHED;
		return false;
	}

	m_prev_err_norm = err_norm;
	m_status = CLMSolver::LM_CALC_J;
	return true;
}

void CLMSolver::lambdaInit(i32 min_lambda, i32 max_lambda)
{
	m_min_lambda = min_lambda;
	m_max_lambda = max_lambda;
}

bool CLMSolver::step(f64* model_param_ptr)
{
	//calc delta matrix by Levenberg-Marquardt
	static const f64 LOG10 = log(10);
	f64 lambda = exp(m_lambda_lg10*LOG10);

	CPOMatrix::copyMatrix(m_matJtJ, m_matJtJDiag);
	CPOMatrix::multiplyDiag(m_matJtJDiag, 1 + lambda);
	if (!CPOMatrix::inverse(m_matJtJDiag, m_matJtJI, PO_DECOMP_LU))
	{
		simplelog("CLMSolver InverseMatrix Fail");
	}
	CPOMatrix::multiply(m_matJtJI, m_matJtErr, m_matD);

	//update param
	f64* param_delta_ptr = (f64*)m_matD.data;
	for (i32 i = 0; i < m_param_count; i++)
	{
		model_param_ptr[i] = m_prev_param_ptr[i] - param_delta_ptr[i];
	}
	return true;
}

//////////////////////////////////////////////////////////////////////////
CGNSolver::CGNSolver(i32 param_count, i32 sample_count, i32 max_iteration, f64 err_threshold)
{
	m_param_count = param_count;
	m_max_iteration = max_iteration;
	m_err_threshold = err_threshold;
	
	m_status = GN_CALC_JERR;
	m_iteration = 0;
	m_prev_err_norm = DBL_MAX;
	m_prev_param_ptr = new f64[param_count];
	memset(m_prev_param_ptr, 0, sizeof(f64)*param_count);

	CPOMatrix::initMatrix(m_matJt, param_count, sample_count, PO_TYPE_F64);
	CPOMatrix::initMatrix(m_matJtJ, param_count, param_count, PO_TYPE_F64);
	CPOMatrix::initMatrix(m_matJtJI, param_count, param_count, PO_TYPE_F64);
	CPOMatrix::initMatrix(m_matJtErr, param_count, 1, PO_TYPE_F64);
	CPOMatrix::initMatrix(m_matD, param_count, 1, PO_TYPE_F64);
}

CGNSolver::~CGNSolver()
{
	POSAFE_DELETE_ARRAY(m_prev_param_ptr);
	CPOMatrix::freeMatrix(m_matJt);
	CPOMatrix::freeMatrix(m_matJtJ);
	CPOMatrix::freeMatrix(m_matJtErr);
	CPOMatrix::freeMatrix(m_matJtJI);
	CPOMatrix::freeMatrix(m_matD);
}

i32 CGNSolver::getStatus()
{
	return m_status;
}

bool CGNSolver::update(f64* model_param_ptr, const PoMat& matJ, const PoMat& matErr, f64 err_norm)
{
	if (!model_param_ptr)
	{
		return false;
	}

	//update param with Jacobian matrix
	CPOMatrix::transpose(matJ, m_matJt);
	CPOMatrix::multiply(m_matJt, matJ, m_matJtJ);
	CPOMatrix::multiply(m_matJt, matErr, m_matJtErr);
	if (!CPOMatrix::inverse(m_matJtJ, m_matJtJI, PO_DECOMP_LU))
	{
		simplelog("CGNSolver InverseMatrix Fail");
		m_status = GN_CALC_FAILED;
		return false;
	}

	CPOMatrix::multiply(m_matJtJI, m_matJtErr, m_matD);
	f64* delta_param_ptr = (f64*)m_matD.data;
	for (i32 i = 0; i < m_param_count; i++)
	{
		model_param_ptr[i] -= delta_param_ptr[i];
	}

	if (++m_iteration > m_max_iteration || err_norm < m_err_threshold ||
		CPOMatrix::norm(m_prev_param_ptr, model_param_ptr, m_param_count, PO_NORM_L2) < m_err_threshold)
	{
		m_status = GN_CALC_FINISHED;
		return false;
	}

	m_prev_err_norm = err_norm;
	CPOBase::memCopy(m_prev_param_ptr, model_param_ptr, m_param_count);
	return true;
}
