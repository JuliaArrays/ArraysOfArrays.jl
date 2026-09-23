# This file is a part of ArraysOfArrays.jl, licensed under the MIT License (MIT).

using StructArrays: StructArray

if !isdefined(Main, :Waveform)
    # Like ArrayOfRDWaveforms of RadiationDetectorSignals: a StructArray whose
    # element type may be less specific than its columns, with a specialized
    # getindex that returns the actual column elements without conversion:
    struct Waveform{TV<:AbstractVector,SV<:AbstractVector}
        time::TV
        signal::SV
    end
    Base.@propagate_inbounds Base.getindex(A::StructArray{<:Waveform}, i::Int) = Waveform(A.time[i], A.signal[i])
end
