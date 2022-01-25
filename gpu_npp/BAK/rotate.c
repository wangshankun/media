#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <FreeImage.h>

#include <cuda_runtime.h>
#include <npp.h>


int main( int argc, char* argv[] )
{
    cudaError_t cuRet;
    NppStatus nppRet;
    BOOL fiRet;
    FIBITMAP* pSrcBmp;
    FIBITMAP* pDstBmp;
    unsigned char* pSrcData;
    unsigned char* pDstData;
    Npp8u* pSrcDataCUDA;
    Npp8u* pDstDataCUDA;
    NppiSize oSrcSize;
    NppiSize oDstSize;
    NppiRect oSrcROI;
    NppiRect oDstROI;
    int nSrcPitch;
    int nDstPitch;
    int nSrcPitchCUDA;
    int nDstPitchCUDA;
    double aBoundingBox[2][2];
    double nAngle;


    /* 设置显卡,构建上下文 */
    cuRet = cudaSetDevice( 0 );
    assert( cuRet == cudaSuccess );

    /* 打开文件 */
    pSrcBmp = FreeImage_Load( FIF_BMP, "1.bmp" );
    assert( pSrcBmp != NULL );

    pSrcData = FreeImage_GetBits( pSrcBmp );
    assert( pSrcData != NULL );

    oSrcSize.width = ( int )FreeImage_GetWidth( pSrcBmp );
    oSrcSize.height = ( int )FreeImage_GetHeight( pSrcBmp );
    nSrcPitch = ( int )FreeImage_GetPitch( pSrcBmp );

    oSrcROI.x = oSrcROI.y = 0;
    oSrcROI.width = oSrcSize.width;
    oSrcROI.height = oSrcSize.height;

    nAngle = 45;


    /* 分配显存 */
    pSrcDataCUDA = nppiMalloc_8u_C1( oSrcSize.width, oSrcSize.height, &nSrcPitchCUDA );
    assert( pSrcDataCUDA != NULL );


    /* 计算旋转后长宽 */
    nppiGetRotateBound( oSrcROI, aBoundingBox, nAngle, 0, 0 );
    oDstSize.width = ( int )ceil( fabs( aBoundingBox[1][0] - aBoundingBox[0][0] ) );
    oDstSize.height = ( int )ceil( fabs( aBoundingBox[1][1] - aBoundingBox[0][1] ) );

    /* 建目标图 */
    pDstBmp = FreeImage_Allocate( oDstSize.width, oDstSize.height, 8 );
    assert( pDstBmp != NULL );

    pDstData = FreeImage_GetBits( pDstBmp );

    nDstPitch = ( int )FreeImage_GetPitch( pDstBmp );
    oDstROI.x = oDstROI.y = 0;
    oDstROI.width = oDstSize.width;
    oDstROI.height = oDstSize.height;

    /* 分配显存 */
    pDstDataCUDA = nppiMalloc_8u_C1( oDstSize.width, oDstSize.height, &nDstPitchCUDA );
    assert( pDstDataCUDA != NULL );

    cudaMemcpy2D( pSrcDataCUDA, nSrcPitchCUDA, pSrcData, nSrcPitch, oSrcSize.width, oSrcSize.height, cudaMemcpyHostToDevice );
    cudaMemset2D( pDstDataCUDA, nDstPitchCUDA, 0, oDstSize.width, oDstSize.height );

    /* 处理 */
    nppRet = nppiRotate_8u_C1R( pSrcDataCUDA, oSrcSize, nSrcPitchCUDA, oSrcROI,
                                pDstDataCUDA, nDstPitchCUDA, oDstROI,
                                nAngle, - aBoundingBox[0][0], - aBoundingBox[0][1], NPPI_INTER_CUBIC );
    assert( nppRet == NPP_NO_ERROR );

    cudaMemcpy2D( pDstData, nDstPitch, pDstDataCUDA, nDstPitchCUDA, oDstSize.width, oDstSize.height, cudaMemcpyDeviceToHost );

    fiRet = FreeImage_Save( FIF_BMP, pDstBmp, "2.bmp" );
    assert( fiRet );

    nppiFree( pSrcDataCUDA );
    nppiFree( pDstDataCUDA );

    cudaDeviceReset();

    FreeImage_Unload( pSrcBmp );
    FreeImage_Unload( pDstBmp );

    return 0;
}