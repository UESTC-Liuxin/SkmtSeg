from modeling.backbone import resnet, xception, drn, mobilenet

def build_backbone(backbone,output_stride, BatchNorm,num_classes):
    if backbone == 'resnet50':
        return resnet.resnet50(output_stride, BatchNorm,num_classes)
    elif backbone == 'resnet101':
        return resnet.resnet101(output_stride, BatchNorm,num_classes)
    elif backbone == 'wide_resnet50_2':
        return resnet.wide_resnet50_2(output_stride, BatchNorm,num_classes)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm)
    else:
        raise NotImplementedError
