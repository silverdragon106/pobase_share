///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
// 2011 Jason Newton <nevion@gmail.com>
//M*/
//
#include <opencv2/opencv.hpp>
#include <vector>
#include "define.h"

//! connected components algorithm output formats
enum MyConnectedComponentsTypes {
	MY_CC_STAT_LEFT = 0, //!< The leftmost (x) coordinate which is the inclusive start of the bounding
	//!< box in the horizontal direction.
	MY_CC_STAT_TOP = 1, //!< The topmost (y) coordinate which is the inclusive start of the bounding
	//!< box in the vertical direction.
	MY_CC_STAT_WIDTH = 2, //!< The horizontal size of the bounding box
	MY_CC_STAT_HEIGHT = 3, //!< The vertical size of the bounding box
	MY_CC_STAT_AREA = 4, //!< The total area (in pixels) of the connected component
	MY_CC_STAT_MAX = 5
};

using namespace cv;
namespace MyConnectedComponents{
    struct NoOp{
        NoOp(){
        }
        void init(i32 /*labels*/){
        }
        inline
        void operator()(i32 r, i32 c, i32 l){
            (void) r;
            (void) c;
            (void) l;
        }
        void finish(){}
    };
    struct Point2ui64{
        uint64 x, y;
        Point2ui64(uint64 _x, uint64 _y):x(_x), y(_y){}
    };

    struct CCStatsOp{
        const _OutputArray* _mstatsv;
        cv::Mat statsv;
        const _OutputArray* _mcentroidsv;
        cv::Mat centroidsv;
        std::vector<Point2ui64> integrals;

        CCStatsOp(OutputArray _statsv, OutputArray _centroidsv): _mstatsv(&_statsv), _mcentroidsv(&_centroidsv){
        }
        inline
        void init(i32 nlabels){
            _mstatsv->create(cv::Size(MY_CC_STAT_MAX, nlabels), cv::DataType<i32>::type);
            statsv = _mstatsv->getMat();
            _mcentroidsv->create(cv::Size(2, nlabels), cv::DataType<f64>::type);
            centroidsv = _mcentroidsv->getMat();

            for(i32 l = 0; l < (i32) nlabels; ++l){
                i32 *row = (i32 *) &statsv.at<i32>(l, 0);
                row[MY_CC_STAT_LEFT] = INT_MAX;
                row[MY_CC_STAT_TOP] = INT_MAX;
                row[MY_CC_STAT_WIDTH] = INT_MIN;
                row[MY_CC_STAT_HEIGHT] = INT_MIN;
                row[MY_CC_STAT_AREA] = 0;
            }
            integrals.resize(nlabels, Point2ui64(0, 0));
        }
        void operator()(i32 r, i32 c, i32 l){
            i32 *row = &statsv.at<i32>(l, 0);
            row[MY_CC_STAT_LEFT] = MIN(row[MY_CC_STAT_LEFT], c);
            row[MY_CC_STAT_WIDTH] = MAX(row[MY_CC_STAT_WIDTH], c);
            row[MY_CC_STAT_TOP] = MIN(row[MY_CC_STAT_TOP], r);
            row[MY_CC_STAT_HEIGHT] = MAX(row[MY_CC_STAT_HEIGHT], r);
            row[MY_CC_STAT_AREA]++;
            Point2ui64 &integral = integrals[l];
            integral.x += c;
            integral.y += r;
        }
        void finish(){
            for(i32 l = 0; l < statsv.rows; ++l){
                i32 *row = &statsv.at<i32>(l, 0);
                row[MY_CC_STAT_WIDTH] = row[MY_CC_STAT_WIDTH] - row[MY_CC_STAT_LEFT] + 1;
                row[MY_CC_STAT_HEIGHT] = row[MY_CC_STAT_HEIGHT] - row[MY_CC_STAT_TOP] + 1;

                Point2ui64 &integral = integrals[l];
                f64 *centroid = &centroidsv.at<f64>(l, 0);
                f64 area = ((unsigned*)row)[MY_CC_STAT_AREA];
                centroid[0] = f64(integral.x) / area;
                centroid[1] = f64(integral.y) / area;
            }
        }
    };

    //Find the root of the tree of node i
    template<typename LabelT>
    inline static
    LabelT findRoot(const LabelT *P, LabelT i){
        LabelT root = i;
        while(P[root] < root){
            root = P[root];
        }
        return root;
    }

    //Make all nodes in the path of node i point to root
    template<typename LabelT>
    inline static
    void setRoot(LabelT *P, LabelT i, LabelT root){
        while(P[i] < i){
            LabelT j = P[i];
            P[i] = root;
            i = j;
        }
        P[i] = root;
    }

    //Find the root of the tree of the node i and compress the path in the process
    template<typename LabelT>
    inline static
    LabelT find(LabelT *P, LabelT i){
        LabelT root = findRoot(P, i);
        setRoot(P, i, root);
        return root;
    }

    //unite the two trees containing nodes i and j and return the new root
    template<typename LabelT>
    inline static
    LabelT set_union(LabelT *P, LabelT i, LabelT j){
        LabelT root = findRoot(P, i);
        if(i != j){
            LabelT rootj = findRoot(P, j);
            if(root > rootj){
                root = rootj;
            }
            setRoot(P, j, root);
        }
        setRoot(P, i, root);
        return root;
    }

    //Flatten the Union Find tree and relabel the components
    template<typename LabelT>
    inline static
    LabelT flattenL(LabelT *P, LabelT length){
        LabelT k = 1;
        for(LabelT i = 1; i < length; ++i){
            if(P[i] < i){
                P[i] = P[P[i]];
            }else{
                P[i] = k; k = k + 1;
            }
        }
        return k;
    }

    //Based on "Two Strategies to Speed up Connected Components Algorithms", the SAUF (Scan array union find) variant
    //using decision trees
    //Kesheng Wu, et al
    //Note: rows are encoded as position in the "rows" array to save lookup times
    //reference for 4-way: {{-1, 0}, {0, -1}};//b, d neighborhoods
    const i32 G4[2][2] = {{1, 0}, {0, -1}};//b, d neighborhoods
    //reference for 8-way: {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}};//a, b, c, d neighborhoods
    const i32 G8[4][2] = {{1, -1}, {1, 0}, {1, 1}, {0, -1}};//a, b, c, d neighborhoods
    template<typename LabelT, typename PixelT, typename StatsOp = NoOp >
    struct LabelingImpl{
    LabelT operator()(const cv::Mat &I, cv::Mat &L, i32 connectivity, StatsOp &sop){
        CV_Assert(L.rows == I.rows);
        CV_Assert(L.cols == I.cols);
        CV_Assert(connectivity == 8 || connectivity == 4);
        const i32 rows = L.rows;
        const i32 cols = L.cols;
        //A quick and dirty upper bound for the maximimum number of labels.  The 4 comes from
        //the fact that a 3x3 block can never have more than 4 unique labels for both 4 & 8-way
        const size_t Plength = 4 * (size_t(rows + 3 - 1)/3) * (size_t(cols + 3 - 1)/3);
        LabelT *P = (LabelT *) fastMalloc(sizeof(LabelT) * Plength);
        P[0] = 0;
        LabelT lunique = 1;
        //scanning phase
        for(i32 r_i = 0; r_i < rows; ++r_i){
            LabelT * const Lrow = L.ptr<LabelT>(r_i);
            LabelT * const Lrow_prev = (LabelT *)(((char *)Lrow) - L.step.p[0]);
            const PixelT * const Irow = I.ptr<PixelT>(r_i);
            const PixelT * const Irow_prev = (const PixelT *)(((char *)Irow) - I.step.p[0]);
            LabelT *Lrows[2] = {
                Lrow,
                Lrow_prev
            };
            const PixelT *Irows[2] = {
                Irow,
                Irow_prev
            };
            if(connectivity == 8){
                const i32 a = 0;
                const i32 b = 1;
                const i32 c = 2;
                const i32 d = 3;
                const bool T_a_r = (r_i - G8[a][0]) >= 0;
                const bool T_b_r = (r_i - G8[b][0]) >= 0;
                const bool T_c_r = (r_i - G8[c][0]) >= 0;
                for(i32 c_i = 0; Irows[0] != Irow + cols; ++Irows[0], c_i++){
                    if(!*Irows[0]){
                        Lrow[c_i] = 0;
                        continue;
                    }
                    Irows[1] = Irow_prev + c_i;
                    Lrows[0] = Lrow + c_i;
                    Lrows[1] = Lrow_prev + c_i;
                    const bool T_a = T_a_r && (c_i + G8[a][1]) >= 0   && *(Irows[G8[a][0]] + G8[a][1]);
                    const bool T_b = T_b_r                            && *(Irows[G8[b][0]] + G8[b][1]);
                    const bool T_c = T_c_r && (c_i + G8[c][1]) < cols && *(Irows[G8[c][0]] + G8[c][1]);
                    const bool T_d =          (c_i + G8[d][1]) >= 0   && *(Irows[G8[d][0]] + G8[d][1]);

                    //decision tree
                    if(T_b){
                        //copy(b)
                        *Lrows[0] = *(Lrows[G8[b][0]] + G8[b][1]);
                    }else{//not b
                        if(T_c){
                            if(T_a){
                                //copy(c, a)
                                *Lrows[0] = set_union(P, *(Lrows[G8[c][0]] + G8[c][1]), *(Lrows[G8[a][0]] + G8[a][1]));
                            }else{
                                if(T_d){
                                    //copy(c, d)
                                    *Lrows[0] = set_union(P, *(Lrows[G8[c][0]] + G8[c][1]), *(Lrows[G8[d][0]] + G8[d][1]));
                                }else{
                                    //copy(c)
                                    *Lrows[0] = *(Lrows[G8[c][0]] + G8[c][1]);
                                }
                            }
                        }else{//not c
                            if(T_a){
                                //copy(a)
                                *Lrows[0] = *(Lrows[G8[a][0]] + G8[a][1]);
                            }else{
                                if(T_d){
                                    //copy(d)
                                    *Lrows[0] = *(Lrows[G8[d][0]] + G8[d][1]);
                                }else{
                                    //new label
                                    *Lrows[0] = lunique;
                                    P[lunique] = lunique;
                                    lunique = lunique + 1;
                                }
                            }
                        }
                    }
                }
            }else{
                //B & D only
                const i32 b = 0;
                const i32 d = 1;
                const bool T_b_r = (r_i - G4[b][0]) >= 0;
                for(i32 c_i = 0; Irows[0] != Irow + cols; ++Irows[0], c_i++){
                    if(!*Irows[0]){
                        Lrow[c_i] = 0;
                        continue;
                    }
                    Irows[1] = Irow_prev + c_i;
                    Lrows[0] = Lrow + c_i;
                    Lrows[1] = Lrow_prev + c_i;
                    const bool T_b = T_b_r                            && *(Irows[G4[b][0]] + G4[b][1]);
                    const bool T_d =          (c_i + G4[d][1]) >= 0   && *(Irows[G4[d][0]] + G4[d][1]);
                    if(T_b){
                        if(T_d){
                            //copy(d, b)
                            *Lrows[0] = set_union(P, *(Lrows[G4[d][0]] + G4[d][1]), *(Lrows[G4[b][0]] + G4[b][1]));
                        }else{
                            //copy(b)
                            *Lrows[0] = *(Lrows[G4[b][0]] + G4[b][1]);
                        }
                    }else{
                        if(T_d){
                            //copy(d)
                            *Lrows[0] = *(Lrows[G4[d][0]] + G4[d][1]);
                        }else{
                            //new label
                            *Lrows[0] = lunique;
                            P[lunique] = lunique;
                            lunique = lunique + 1;
                        }
                    }
                }
            }
        }

        //analysis
        LabelT nLabels = flattenL(P, lunique);
        sop.init(nLabels);

        for(i32 r_i = 0; r_i < rows; ++r_i){
            LabelT *Lrow_start = L.ptr<LabelT>(r_i);
            LabelT *Lrow_end = Lrow_start + cols;
            LabelT *Lrow = Lrow_start;
            for(i32 c_i = 0; Lrow != Lrow_end; ++Lrow, ++c_i){
                const LabelT l = P[*Lrow];
                *Lrow = l;
                sop(r_i, c_i, l);
            }
        }

        sop.finish();
        fastFree(P);

        return nLabels;
    }//End function LabelingImpl operator()

    };//End struct LabelingImpl
}//end namespace connectedcomponents

//L's type must have an appropriate depth for the number of pixels in I
template<typename StatsOp>
static
i32 myConnectedComponents_sub1(const cv::Mat &I, cv::Mat &L, i32 connectivity, StatsOp &sop){
    CV_Assert(L.channels() == 1 && I.channels() == 1);
    CV_Assert(connectivity == 8 || connectivity == 4);

    i32 lDepth = L.depth();
    i32 iDepth = I.depth();
    using MyConnectedComponents::LabelingImpl;
    //warn if L's depth is not sufficient?

    CV_Assert(iDepth == CV_8U || iDepth == CV_8S);

    if(lDepth == CV_8U){
        return (i32) LabelingImpl<uchar, uchar, StatsOp>()(I, L, connectivity, sop);
    }else if(lDepth == CV_16U){
        return (i32) LabelingImpl<ushort, uchar, StatsOp>()(I, L, connectivity, sop);
    }else if(lDepth == CV_32S){
        //note that signed types don't really make sense here and not being able to use unsigned matters for scientific projects
        //OpenCV: how should we proceed?  .at<T> typechecks in debug mode
        return (i32) LabelingImpl<i32, uchar, StatsOp>()(I, L, connectivity, sop);
    }

    CV_Error(CV_StsUnsupportedFormat, "unsupported label/image type");
    return -1;
}

i32 myConnectedComponents(InputArray _img, OutputArray _labels, i32 connectivity, i32 ltype){
    const cv::Mat img = _img.getMat();
    _labels.create(img.size(), CV_MAT_DEPTH(ltype));
    cv::Mat labels = _labels.getMat();
    MyConnectedComponents::NoOp sop;
    if(ltype == CV_16U){
        return myConnectedComponents_sub1(img, labels, connectivity, sop);
    }else if(ltype == CV_32S){
        return myConnectedComponents_sub1(img, labels, connectivity, sop);
    }else{
        CV_Error(CV_StsUnsupportedFormat, "the type of labels must be 16u or 32s");
        return 0;
    }
}

i32 myConnectedComponentsWithStats(InputArray _img, OutputArray _labels, OutputArray statsv,
                                     OutputArray centroids, i32 connectivity, i32 ltype)
{
    const cv::Mat img = _img.getMat();
    _labels.create(img.size(), CV_MAT_DEPTH(ltype));
    cv::Mat labels = _labels.getMat();
    MyConnectedComponents::CCStatsOp sop(statsv, centroids);
    if(ltype == CV_16U){
        return myConnectedComponents_sub1(img, labels, connectivity, sop);
    }else if(ltype == CV_32S){
        return myConnectedComponents_sub1(img, labels, connectivity, sop);
    }else{
        CV_Error(CV_StsUnsupportedFormat, "the type of labels must be 16u or 32s");
        return 0;
    }
}
