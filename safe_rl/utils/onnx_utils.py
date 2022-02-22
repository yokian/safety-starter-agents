import onnxruntime as ort

def get_model_lambda(fpath, output_index=None):
    """ Returns lambda fn to get output of the onnx network given a torch tensor
        Needs onnxruntime 
        
        Args:
            fpath: path to onnx model
            output_index: None for all outputs and -1 for 'pi' output

        Usage:
            pi = get_model_lambda('point_cpo.onnx', -1)
            x = torch.zeros([1,30])
            mean, std = pi(x)
        
        """
    ort_sess = ort.InferenceSession(fpath)
    if output_index is not None:
        return lambda xt: ort_sess.run(None,{'x':xt.float().numpy()})[output_index]
    else:
        return lambda xt: ort_sess.run(None,{'x':xt.float().numpy()})