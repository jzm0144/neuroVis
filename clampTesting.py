


# ------------------------------  Seeing the Differences between  ------------------------
# ------------------------------  PTSD and Control Clamping       ------------------------

for heatmap in heatmaps:
    #Create the analyzers
    analyzer = innvestigate.create_analyzer(heatmap[0],
                                            model_no_softmax,
                                            neuron_selection_mode = "index",
                                            **heatmap[1])

    # Generate the heatmaps
    analysis_ctrl_clamp = analyzer.analyze(inputs, 0)
    analysis_ptsd_clamp = analyzer.analyze(inputs, 1)

    # Delete the explanations where prediction was incorrect
    m = []
    for __ in range(analysis_ctrl_clamp.shape[0]):
        predNeuron=np.argmax(preds[__]),
        actualNeuron=np.argmax(outs[__])
        if predNeuron != actualNeuron:
            m.append(__)
            print("skipped = ", __)
    analysis_ctrl_clamp = np.delete(analysis_ctrl_clamp, m, 0)
    analysis_ptsd_clamp = np.delete(analysis_ptsd_clamp, m, 0)
    print(m)
    del m
    # Working with 8 Control Subjects
    controls_with_ctrl_clamp  = analysis_ctrl_clamp[:8,:,:,0]
    controls_with_ptsd_clamp  = analysis_ptsd_clamp[:8,:,:,0]



    
    groupItems = controls_with_ctrl_clamp.shape[0]

    comb8c = list(combinations([_ for _ in range(8)], 2))

    control_ctrl_clamp_Dist = 0
    control_ptsd_clamp_Dist = 0
    bwgroupsDist = 0
    for _ in comb8c:
        i, j = _

        control_ctrl_clamp_Dist  += np.sqrt(np.square(controls_with_ctrl_clamp[i] - controls_with_ctrl_clamp[j]))
        control_ptsd_clamp_Dist  += np.sqrt(np.square(controls_with_ptsd_clamp[i] - controls_with_ptsd_clamp[j]))
        bwgroupsDist             += np.sqrt(np.square(controls_with_ctrl_clamp[i] - controls_with_ptsd_clamp[j]))

    print("Distance Within Controls    = ", np.sum(control_ctrl_clamp_Dist/len(comb8c)))
    print("Distance Within PTSD        = ", np.sum(control_ptsd_clamp_Dist/len(comb8c)))
    print("Distance b/w PTSD & Control = ", np.sum(bwgroupsDist/len(comb8c)))
    
    plt.figure(1)
    plt.subplot(331)
    m1 = sb.heatmap(control_ctrl_clamp_Dist/len(comb8c))
    plt.subplot(335)
    m2 = sb.heatmap(control_ptsd_clamp_Dist/len(comb8c))
    plt.subplot(339)
    m3 = sb.heatmap(bwgroupsDist/len(comb8c))
    plt.savefig('dist_control.png')
    
    # Working with 8 PTSD Subjects
    ptsd_with_ctrl_clamp  = analysis_ctrl_clamp[8:16,:,:,0]
    ptsd_with_ptsd_clamp  = analysis_ptsd_clamp[8:16,:,:,0]

    
    groupItems = ptsd_with_ctrl_clamp.shape[0]

    comb8c = list(combinations([_ for _ in range(8)], 2))

    ptsd_ctrl_clamp_Dist = 0
    ptsd_ptsd_clamp_Dist = 0
    bwgroupsDist = 0
    for _ in comb8c:
        i, j = _

        ptsd_ctrl_clamp_Dist  += np.sqrt(np.square(ptsd_with_ctrl_clamp[i] - ptsd_with_ctrl_clamp[j]))
        ptsd_ptsd_clamp_Dist  += np.sqrt(np.square(ptsd_with_ptsd_clamp[i] - ptsd_with_ptsd_clamp[j]))
        bwgroupsDist          += np.sqrt(np.square(ptsd_with_ctrl_clamp[i] - ptsd_with_ptsd_clamp[j]))

    print("Distance Within Controls    = ", np.sum(ptsd_ctrl_clamp_Dist/len(comb8c)))
    print("Distance Within PTSD        = ", np.sum(ptsd_ptsd_clamp_Dist/len(comb8c)))
    print("Distance b/w PTSD & Control = ", np.sum(bwgroupsDist/len(comb8c)))

    plt.subplot(331)
    m1 = sb.heatmap(ptsd_ctrl_clamp_Dist/len(comb8c))
    plt.subplot(335)
    m2 = sb.heatmap(ptsd_ptsd_clamp_Dist/len(comb8c))
    plt.subplot(339)
    m3 = sb.heatmap(bwgroupsDist/len(comb8c))
    plt.savefig('dist_ptsd.png')