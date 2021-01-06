<table>
    <tr>
    	<td style="width:6%">组别</td>
        <td style="width:28%">测试图片</td>
        <td style="width:28%">预处理结果</td>
        <td style="width:28%">求解结果</td>
        <td width="10%">备注说明</td>
    </tr>
    <tr>
        <td>1-1</td>
        <td><img src="../test1/1-1.jpg"/></td>
        <td><img src="../result/1-1_proc.png"/></td>
        <td><img src="../result/1-1_sol.png"/></td>
        <td>识别效果非常差</td>
    </tr>
    <tr>
                <td>1-2</td>
                <td><img src="../test1/1-2.jpg"/></td>
                <td><img src="../result/1-2_proc.png"/></td>
                <td><img src="../result/1-2_sol.png"/></td>
                <td>识别效果非常差</td>
            </tr>
            <tr>
                <td>1-3</td>
                <td><img src="../test1/1-3.jpg"/></td>
                <td><img src="../result/1-3_proc.png"/></td>
                <td><img src="../result/1-3_sol.png"/></td>
                <td>识别效果非常差</td>
            </tr>
            <tr>
                <td>1-4</td>
                <td><img src="../test1/1-4.jpg"/></td>
                <td><img src="../result/1-4_proc.png"/></td>
                <td><img src="../result/1-4_sol.png"/></td>
                <td>识别效果非常差</td>
            </tr>
            <tr>
                <td>1-5</td>
                <td><img src="../test1/1-5.jpg"/></td>
                <td><img src="../result/1-5_proc.png"/></td>
                <td><img src="../result/1-5_sol.png"/></td>
                <td>识别效果非常差</td>
            </tr>
            <tr>
                <td>2-1</td>
                <td><img src="../test1/2-1.jpg"/></td>
                <td><img src="../result/2-1_proc.png"/></td>
                <td><img src="../result/2-1_sol.png"/></td>
                <td>全部识别正确，求解无误，本地测得解数独耗时 32ms</td>
            </tr>
            <tr>
                <td>2-2</td>
                <td><img src="../test1/2-2.jpg"/></td>
                <td><img src="../result/2-2_proc.png"/></td>
                <td><img src="../result/2-2_sol.png"/></td>
                <td>全部识别正确，求解无误，本地测得解数独耗时 36ms</td>
            </tr>
            <tr>
                <td>2-3</td>
                <td><img src="../test1/2-3.jpg"/></td>
                <td><img src="../result/2-3_proc.png"/></td>
                <td><img src="../result/2-3_sol.png"/></td>
                <td>仅一处识别错误，且是因图像预处理质量不理想导致。误将(3,6)处的七识别为二，导致不可解。</td>
            </tr>
            <tr>
                <td>2-4</td>
                <td><img src="../test1/2-4.jpg"/></td>
                <td><img src="../result/2-4_proc.png"/></td>
                <td><img src="../result/2-4_sol.png"/></td>
                <td>仅两处由图像预处理质量不理想导致的识别错误。误将(2,6)处的五识别为二，导致不可解。漏认(4,5)处的九。</td>
            </tr>
            <tr>
                <td>2-5</td>
                <td><img src="../test1/2-5.jpg"/></td>
                <td><img src="../result/2-5_proc.png"/></td>
                <td><img src="../result/2-5_sol.png"/></td>
                <td>正经的无解。一处因图像预处理质量不理想导致的漏识别((1,3)处的九)。但这不影响无解的判定，说明无解判定的功能没问题。</td>
            </tr>
            <tr>
                <td>自己加的测试用例</td>
                <td><img src="../test_files/test4.jpg"/></td>
                <td><img src="../result/my_proc.png"/></td>
                <td><img src="../result/my_sol.png"/></td>
                <td>中文、阿拉伯数字混合。正常识别。正确求解。这组测试用例的图片位置正，预处理和识别难度较低。但能说明该模型能正常工作。注：因自己的测例和助教的测例的笔触差异大，故预处理的操作略有不同。</td>
            </tr>
</table>

