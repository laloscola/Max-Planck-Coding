'''
% ****************************************************************************
% FUNCTION:      [y, x, S] = RTxReadBin(filename)
%                [y, x, S] = RTxReadBin(filename, acquisitions, xInterval,...
%                            nOutputSamplesMax)
%
% SPECIFICATION: The R&S RTO series oscilloscopes use a proprietary .bin format
%                as the standard export format for waveform data. 
%
%                The format consists of two files. The waveform samples are
%                stored in the file <filename>.Wfm.bin following the IEEE 754
%                standard for floating point arithmetic. A companion file called
%                <filename>.bin contains the waveform properties in a
%                XML-compatible format.
%   
%                The function RTOReadBin reads binary files exported with 
%                .bin format from the RTO and produces the sample array y, 
%                the horizontal vector x, and the structure S with waveform 
%                properties.
%
% PARAMETERS:    filename - Name of the file. File extension is not
%                required. Path name muss be included if the file is not 
%                located in the current working directory.
%
% OPTIONAL PAR.: acquisitions - There are three possibilities for this input
%                       parameter:
%                       1) If omitted or empty: all available acquisitions 
%                           are read.
%                       2) Scalar value: the corresponding acquisition 
%                           (only one) is read using MATLAB Indexing s.
%                           below
%                       3) Acquisition interval [acqMin, acqMax]:
%                           acquisitions from acqMin until acqMax are read.
%                           i.e. [2 4] means acquisition 2, 3 and 4 are
%                           read.
%
%                           Equivalence between RTO and MATLAB index system
%                           Matlab indexing is used in this function:
%                           RTO Indexing      -6  -5  -4  -3  -2  -1   0
%                                             ---|---|---|---|---|---|---
%                           MATLAB Indexing    1   2   3   4   5   6   7
%                           In the figure above the last waveform acquired
%                           in the RTO corresponds to the index 7
%
%                xInterval - User-defined time/frequency interval [xMin, xMax].
%                       Only samples within this interval are read from the
%                       .bin file. The whole waveform is read when this
%                       parameter is omitted or empty.
%                nOutputSamplesMax - Maximum number of output samples per 
%                       acquistion or xInterval. When the number of samples 
%                       within one acquisition (respectively xInterval)
%                       exceeds nOutputSamplesMax, then the samples
%                       are decimated by the smallest integer that yields a 
%                       number of output samples below nOutputSamplesMax.
%                nNofChannels - RTO supports multiple channel export.
%                   1) If omitted or empty: just one channel is assumed 
%                      (no multi channel export)
%                   2) Scalar value: 1<= nNofChannels <= 4
%                   Example: Reading a multi channel export (2 channels)
%                   [y, x, S] = RTOReadBin(filename, [], [], [], 2);
%
% RETURN VALUES: y - Matrix of waveform samples (double format or for raw 
%                    ADC values - 8 bit integer). 
%                    Each column corresponds to one acquisition.
%                    Last column corresponds to the last acquisition.
%                    In case of multi channel export the variable y is a 
%                    3-dimensional array.
%                    y[NofSamples, NofAcq, NofChs]
%                    i.e: y[5000,1,4] indicates a multi channel export (all
%                    4 analog channels), one acquisition pro channel with
%                    5000 values.
%                    For Envelope the NofChs is double because the resulting 
%                    diagram consists of two waveforms: the minimums
%                    (floor) and maximus (roof). i.e. y[1000,1,2] for one
%                    channel export indicates y(1000,1,1) is the waveform
%                    floor and y(1000,1,2) is the waveform roof.
%                x - Vector of time/frequency instants. Remains unchanged for
%                    all acquisitions.
%                S - Structure with waveform properties.
%
% LIMITATIONS:   Multi-Channel-Export:
%                   nNofChannels must coincide with the number of channels 
%                   exported in the file, which means for four-channel 
%                   export it is not possible to read two of them. For this 
%                   case those channels can be excluded outside this
%                   function.
%                Export using Envelope:
%                   For exports using trace arithmetic envelope, this script 
%                   supports only the case when all channels (in case of multi
%                   channel export) are exported with envelope. 
%                   i.e. two channel export, Ch1 without waveform arithmetic 
%                   and Ch2  with envelope is not supported in this script.
%
% EXCEPTIONS:    Not used.
% ****************************************************************************

% ****************************************************************************
%
% Copyright: (c) 2013 Rohde & Schwarz. All rights reserved.
%                Muehldorfstr. 15, D-81671 Munich, Germany
%
% TERMS OF USE:  Rohde & Schwarz products and services are supplied to
%                customers subject to certain contractual terms and 
%                conditions. In addition, there are some requirements that 
%                apply especially to certain products, customers or 
%                circumstances.
%                see http://www.termsofuse.rohde-schwarz.com/
%
% PROJECT:       Python tools.
%
% COMPILER:      Matlab 8.2 (R2013b).
%
% LANGUAGE:      Python Interpreter.
%
% AUTHOR:        Mathias Hellwig
%
% PREMISES:      None.
%
% REMARKS:       None.
%
% HISTORY:       $Log: $
%
% COMMENTS:      RTOReadBin V. 2.0 supports multiple acquisition and 
%                multiple channel export. 
%
% VERSION:       2.0
%
% ****************************************************************************
'''
from logging import Logger
from typing import Dict, List, Union

import numpy as np
import sys
import errno
import os
import logging
from xml.dom import minidom


def RTxReadBin(filename, acquisitions=None, xInterval=None):
    logger: Logger = logging.getLogger(__name__)
    # due to code compatibility with the MATLAB module, this was left in the code
    nOutputSamplesMax = None
    # multi channel wfm files are detected automatically and a warning is asserted, opposed to the MATLAB module
    nNofChannels = 1

    if not os.path.isfile(filename) and not os.path.isfile(filename + '.bin'):
        logger.error('No Inputfile found (%s)' % filename)
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)

    pathstr = os.path.dirname(filename)
    if not pathstr:
        pathstr = '.'
    fname, ext = os.path.splitext(os.path.basename(filename))
    # test for Wfm.bin
    f1name, ext1 = os.path.splitext(fname)
    if ext1 == '.Wfm':
        fname = f1name

    if not ext:
        filenameHeader = filename + '.bin'
        filenameSamples = filename + '.Wfm.bin'
    else:
        filenameHeader = os.path.join(pathstr, ''.join([fname, '.bin']))
        filenameSamples = os.path.join(pathstr, ''.join([fname, '.Wfm.bin']))

    if not os.path.isfile(filenameSamples):
        logger.error('No Samplefile found (%s)' % filenameSamples)
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filenameSamples)

    if not os.path.isfile(filenameHeader):
        logger.warning('No Headerfile found (%s), trying with defaults' % filenameHeader)
        filenameHeader = None

    #    Check validity/existence of nNofChannels
    if not isinstance(nNofChannels, int) or nNofChannels < 1 or nNofChannels > 4:
        logger.error(
            "The parameter nNofChannels must be a positive integer between 1 and 4, respectively 1 and 2. Multiple "
            "channels export is only supported for analog channels")
        raise ValueError('channel number out of scope')

        #   Check validity of horizontal interval
    if xInterval:
        if len(xInterval) != 2 or xInterval[1] < xInterval[0]:
            logger.error('Please specify the parameter xInterval as [xMin, xMax]')
            raise ValueError('illegal interval definition')
    #   Check validity of maximum number of output samples
    if nOutputSamplesMax and not isinstance(nOutputSamplesMax, int) and nOutputSamplesMax < 1:
        logger.error('The parameter nOutputSamplesMax must be a positive integer')
        raise ValueError('illegal sample definition')

    # --- Load waveform parameters from file ---

    # Use default parameters if the previous file check has failed

    if os.path.getsize(filenameHeader) == 0:
        S = {'RecordLength': None,
             'XStart': None,
             'XStop': None}
    else:
        xml = minidom.parse(filenameHeader)
        S = {}
        node = xml.getElementsByTagName('Prop')  # fixed
        nNofChannelsFlag = False
        channelList = []
        MultiChannelAttr = {
            "MultiChannelVerticalOffset": 1,
            "MultiChannelExportState": 0,  # setup the channel list
            "MultiChannelVerticalScale": 1,
            "MultiChannelVerticalPosition": 1
        }
        for nidx in node:
            attrName = nidx.getAttribute('Name')
            attrValue = nidx.getAttribute('Value')
            # support multichannel
            if attrName == 'MultiChannelExport' and 'ONOFF_ON' in attrValue:
                nNofChannelsFlag = True
            if nNofChannelsFlag and attrName in MultiChannelAttr:
                attrNofChannelsMax = int(nidx.getAttribute('Size'))
                tmpList = []
                for idx in range(attrNofChannelsMax):
                    attrStr = 'I_%1d' % idx  # should work up to an 8 ch scope
                    if 'ONOFF_ON' in nidx.getAttribute(attrStr):
                        channelList.append(idx)
                    if MultiChannelAttr[attrName] == 1:
                        tmpList.append(float(nidx.getAttribute(attrStr)))
                if MultiChannelAttr[attrName] == 0:
                    attrValue = channelList
                else:
                    attrValue = tmpList

                if not MultiChannelAttr[attrName] and nNofChannels != len(channelList):
                    channelListStr: str = ', '.join([str(chidx + 1) for chidx in channelList])
                    logger.warning('Found multichannel File, with channels: %s. Using these!' % channelListStr)
                    nNofChannels = len(channelList)
                # else:
                #     logger.info('Found multichannel File, with channels: %s' % channelListStr)

            if isinstance(attrValue, list):
                S[attrName] = attrValue
            elif is_num(attrValue):
                S[attrName] = float(attrValue)
            else:
                S[attrName] = attrValue

    '''
        Get header information from waveform file and finalize the initialization 
        of the parameter structure
        Open the file and read the first eight bytes
        
        Read header segment information. The second value is the number of samples
        in one acquisition. For files with multiple channels or
        acquisitions, this in not an indication of the file length.
    '''
    with open(filenameSamples, 'rb') as fid:
        fileInfo = np.fromfile(fid, np.uint32, 2)
        fid.close()

    if os.path.getsize(filenameHeader) == 0:
        # Initialice the fields that were left empty
        S['SignalHardwareRecordLength'] = fileInfo(1)
        S['SignalResolution'] = 1
        S['XStart'] = 0
        S['XStop'] = fileInfo(1) - 1  # changed compered to the original script
        S['HardwareXStart'] = S['XStart']
        S['HardwareXStop'] = S['XStop']
        nNofSamplesPerAcqPerCh = S['SignalHardwareRecordLength']
    else:
        '''
        For compatibility reasons with FW versions older than 2.0 we need to check 
        that the number of samples in waveform and parameter files match.
        With the release of FW version 2.0 the firmware version is available 
        in the header file. According to this we decide for one of them for the 
        following computations in case the parameters differ from each other.
        '''
        if S['SignalHardwareRecordLength'] != fileInfo[1]:
            logger.warning('Number of samples in waveform file and properties file do not match')
            if 'FirmwareVersion' in S:
                nNofSamplesPerAcqPerCh = S['SignalHardwareRecordLength']  # FW >= 2.0
            else:
                nNofSamplesPerAcqPerCh = fileInfo[1] / nNofChannels  # FW < 2.0
        else:
            nNofSamplesPerAcqPerCh = S['SignalHardwareRecordLength']
    # --- Check for waveform arithmetic ENVELOPE ---
    # Using waveform arithmetic Envelope two envelope waveforms are exported,
    # because of that the number of channels is double internally

    if 'TraceType' in S and 'ENVELOPE' in S['TraceType']:
        if nNofChannels == 1:
            if fileInfo[0] < 2:
                S['MultiChannelExport'] = 'eRS_ONOFF_ON'
                S['MultiChannelExportState'] = 'eRS_ONOFF_ON'
                # now double also the list for the raw format!
                S['MultiChannelVerticalOffset'] = [S['VerticalOffset'], S['VerticalOffset']]
                S['MultiChannelVerticalScale'] = [S['VerticalScale'], S['VerticalScale']]
                S['MultiChannelVerticalPosition'] = [S['VerticalPosition'], S['VerticalPosition']]
                channelList = [0, 1]
        else:
            logger.error('multichannel envelope currently not supported')
            raise NotImplementedError("envelope with multi channel")
                # for idx in range(nNofChannels):
                #     S['MultiChannelVerticalOffset'].insert(2 * idx, S['MultiChannelVerticalOffset'][2 * idx])
                #     S['MultiChannelVerticalScale'].insert(2 * idx, S['MultiChannelVerticalScale'][2 * idx])
                #     S['MultiChannelVerticalPosition'].insert(2 * idx, S['MultiChannelVerticalPosition'][2 * idx])
        nNofChannels = nNofChannels * 2

    '''
        --- Following parameters must be calculated ---

        a) Preamble length
        b) Sample interval
        c) Sample decimation
    '''

    # a) Compute preamble length

    # %   The sample file may have some preamble samples. This samples are needed to reproduce measurements performed
    # on the waveform

    if S['LeadingSettlingSamples']:
        nPreambleSamples = S['LeadingSettlingSamples']
    else:
        # Estimate number of preamble samples(backwards compatibility)
        nPreambleSamples = round((S['XStart'] - S['HardwareXStart']) / S['SignalResolution'])

    if nPreambleSamples < 0:
        logger.warning('Error while computing number of preamble samples')
        nPreambleSamples = 0

    # ---  b) Compute sample interval ---

    # The first valid sample has the index 1 (Matlab indexing) Map the user defined sample interval into a sample
    # index range

    tolerance = 1e-15  # Used for comparing floating values (equals 1 fs for time domain)

    if not xInterval:
        # Default range
        idxRange = np.array([1, S['SignalRecordLength']])
    else:
        idxRange = np.zeros(2)
        if (S['XStart'] - xInterval[0] > tolerance) and (xInterval[1] - S['XStop'] > tolerance):
            logger.warning('User-defined range lies outside waveform horizontal range')
        elif (S['XStart'] - xInterval[0]) > tolerance:
            logger.warning('User-defined lower range lies outside waveform horizontal range')
        elif (xInterval[1] - S['XStop']) > tolerance:
            logger.warning('User-defined upper range lies outside waveform horizontal range')
        else:
            # Compute lower and upper index range
            idxRange[0] = max(1, round((xInterval[0] - S['XStart']) / S['SignalResolution']))
            idxRange[1] = min(S['SignalRecordLength'], round((xInterval[1] - S['XStart']) / S['SignalResolution']))

    # --- c) Compute sample decimation ---
    #   Read a decimated version of the waveform if the user constrains the number of imported samples
    if not nOutputSamplesMax:
        # Read all data
        decimationFactor = 1
        nSamples = int(idxRange[1] - idxRange[0] + 1)
    elif (idxRange[1] - idxRange[0]) <= nOutputSamplesMax:
        # Number ofsamples in range is less than upper limit No decimation required, read all data
        decimationFactor = 1
        nSamples = int(idxRange[1] - idxRange[0] + 1)
    else:
        # Decimation  required
        decimationFactor = np.ceil((idxRange[1] - idxRange[0] + 1) / nOutputSamplesMax)
        nSamples = int(np.floor((idxRange[1] - idxRange[0]) / decimationFactor) + 1)

    # --- Open file and determine size and precision per sample  ---

    with open(filenameSamples, 'rb') as fid:
        # was already tested

        # Get number of bytes and precision per sample
        dataType = fileInfo[0]
        # if dataType == 6:
        #     sampleSizeX = 8
        # else:
        #     sampleSizeX = 0 # samplSizeX is different from zero only for the X/Y interleaved export

        dataTypeDict: Dict[int, List[Union[int, str]]] = {
            0: [0, 1, '', 'i1'],  # y samples are int8
            1: [0, 2, '', 'i2'],  # y samples are int16
            2: [0, 3, '', 'i3'],  # y samples are int24
            3: [0, 4, '', 'i4'],  # y samples are int32
            4: [0, 4, '', 'f4'],  # y samples are single (float)
            5: [0, 8, '', 'f8'],  # y samples are double
            6: [8, 4, 'f8', 'f4'],  # samples are XY interleaved - X: double, Y:float
            7: [0, 1, '', 'u1'],  # MSO data, treated initially as unsigned char and converted later to bin
            8: [0, 4, '', 'f4']  # MSO bus (parbus), written out as float
        }

        if not dataTypeDict[dataType]:
            logger.error('unkown sample data type in waveform file')
            raise NotImplementedError("data type not implemented")

        if 'ONOFF_ON' in S['TimestampState']:
            timestampkey = 'Timestamps'
            timeStampSize = 8  # in byte float64
            logger.info('Creating time stamp list in wfm description : key = "%s"' % timestampkey)
        else:
            timeStampSize = 0

        # Define size and precision for digital waveforms
        # --- Check for digital waveforms ---

        if S['SourceType'] and 'SOURCE_TYPE_DIGITAL' in S['SourceType']:
            bIsDigitalSource = True
            dataType = len(dataTypeDict) - 2
            sampleSizeY = 1
            # adjust nSamples to binary data
            bitpack = 8
            outputFmt = np.bool
        elif S['SourceType'] and 'SOURCE_TYPE_MSO_BUS' in S['SourceType']:
            bIsDigitalSource = False
            dataType = len(dataTypeDict) - 1
            sampleSizeY = 1
            # adjust nSamples to binary data
            bitpack = 1
            outputFmt = np.int16
        else:
            outputFmt = np.float32
            bitpack = 1
            bIsDigitalSource = False

        #  --- Check acquisitions ---

        # % Define constant
        headerSegmentInBytes = 8
        # % Get size (in Bytes) of samples file (<filename>.Wfm.bin)
        # dirInfo = dir(pathstr);
        # [~, name, ~] = fileparts(filenameSamples);
        fileSizeSamples = os.path.getsize(filenameSamples)
        # % Compute number of available acquisitions in file
        if bIsDigitalSource:
            # MSO signals are saved bitwise. Each byte on file corresponds to 8 sequential digital values
            nNofAvailableAcq = np.floor((fileSizeSamples - headerSegmentInBytes) * bitpack /
                                        (S['SignalRecordLength'] * sampleSizeY * nNofChannels))
            # Interleaved x/y is not available for MSO channels
        else:
            nNofAvailableAcq = np.floor((fileSizeSamples - headerSegmentInBytes) /
                                        (S['SignalRecordLength'] * (dataTypeDict[dataType][0] + dataTypeDict[dataType][
                                            1] * nNofChannels) +
                                         timeStampSize))

        if not acquisitions:
            #  No precedent acquisitions to be skiped
            NofPreAcq = 0
            # Set number of acquisitions to the number of available acq. in file
            nNofAcquisitions = int(nNofAvailableAcq)
        elif not isinstance(acquisitions, list) and isinstance(acquisitions, int):
            # Verify acquisition index
            if acquisitions < 1:
                logger.error('Acquistion index must be positive integer')
                raise ValueError('illegal acqisition specifier')
            elif acquisitions > nNofAvailableAcq:
                logger.error(
                    f'Acquisition index exceeds {nNofAvailableAcq:d} which is the biggest acquisition index available')
                raise ValueError('illegal acqisition specifier')
            else:
                # Determine number of precedent acquisitions
                NofPreAcq = acquisitions - 1
                # Just one acquisition will be read.
                nNofAcquisitions = 1
        elif len(acquisitions) == 2:  # Acquisitions betwenn [acqMin, acqMax] will be read
            #     Verify acquisition interval
            if not isinstance(acquisitions[0], int) or acquisitions[0] < 0 or \
                    not isinstance(acquisitions[1], int) or acquisitions[1] < 1:
                logger.error('Values for acquisition interval must be positive integers')
                raise ValueError('illegal acqisition specifier')
            elif acquisitions[0] > acquisitions[1]:
                logger.error('Please specify acquisition interval in the form [acqMin, acqMax]')
                raise ValueError('illegal acqisition specifier')
            elif acquisitions[1] - 1 > nNofAvailableAcq:
                logger.error(
                    "Acquisition interval is out of range. For the current file the biggest acquisition index "
                    "available is %d" % nNofAvailableAcq)
                raise ValueError('illegal acqisition specifier')
            #     % Determine number of precedent acquisitions
            NofPreAcq = acquisitions[0] # - 1
            # Compute number of acquistions to be read
            nNofAcquisitions = int(acquisitions[1] - 1 - acquisitions[0])
        else:
            logger.error(
                'Please specify the parameter acquisition either as a scalar value or as an interval [acqMin, acqMax]')
            raise ValueError('illegal acqisition specifier')

        # --- Verify that the required number of bytes are available in file ---

        # Define constants for skipping preamble and postamble
        nPreambleInBytes = int((nPreambleSamples + idxRange[0] - 1) *
                               (dataTypeDict[dataType][0] + dataTypeDict[dataType][1] * nNofChannels))
        nPostambleInBytes = int(
            (nNofSamplesPerAcqPerCh - nSamples * decimationFactor - nPreambleSamples - (idxRange[0] - 1)) *
            (dataTypeDict[dataType][0] + dataTypeDict[dataType][1] * nNofChannels))

        # Number of samples given in byte for all channels
        nNofBytes_AllCHs = nSamples / bitpack * (dataTypeDict[dataType][0] + dataTypeDict[dataType][1] * nNofChannels) + \
                           timeStampSize + nPreambleInBytes + nPostambleInBytes

        # Number of samples given in byte for all acquisitions and all channels
        nNofBytes_AllCHs_AllAcqs = headerSegmentInBytes + nNofBytes_AllCHs * nNofAcquisitions

        # Check if the number of required bytes are available in file
        # if nNofBytes_AllCHs_AllAcqs > fileSizeSamples and not bIsDigitalSource or \
        #     nNofBytes_AllCHs_AllAcqs > fileSizeSamples and bIsDigitalSource:
        if nNofBytes_AllCHs_AllAcqs > fileSizeSamples:
            logger.error('Number of required samples exceeds number of samples available in file')
            raise ValueError('insufficient data')

        # --- Skip 8 byte header ---
        fid.seek(headerSegmentInBytes)

        # --- Skip precedent acquisitions (if necessary) ---
        # 2nd parameter indicates relative movement
        fid.seek(int((nNofSamplesPerAcqPerCh * NofPreAcq) *
                     (dataTypeDict[dataType][0] + dataTypeDict[dataType][1] * nNofChannels)), 1)

        # skip previous timestamps if necessary
        if 'ONOFF_ON' in S['TimestampState']:
            fid.seek(8 * NofPreAcq, 1)
            timestamps = np.zeros((nNofAcquisitions,), np.float64)

        # --- Read waveform samples ---

        # Initialize matrix or three dimensional array of waveform samples
        y = np.zeros((nSamples, nNofAcquisitions, nNofChannels), dtype=outputFmt)

        if dataType < 2:  # either int8 or int 16
            # need to add the singel raw handling!
            VerticalDivisionCount = int(S['VerticalDivisionCount'])
            NofQuantisationLevels = int(S['NofQuantisationLevels'])
            # NofQuantisationLevels = 253 * (256 ** (dataTypeDict[dataType][1] - 1))
            if S['MultiChannelExportState']:
                # channelList
                VerticalOffset = S['MultiChannelVerticalOffset']
                VerticalScale = S['MultiChannelVerticalScale']
                VerticalPosition = S['MultiChannelVerticalPosition']
            else:
                channelList = [0]
                VerticalOffset = [S['VerticalOffset']]
                VerticalScale = [S['VerticalScale']]
                VerticalPosition = [S['VerticalPosition']]

        # nPostambleInBytes = (nNofSamplesPerAcqPerCh - nSamples*decimationFactor - nPreambleSamples - (idxRange(
        # 1)-1)) * (sampleSizeX + sampleSizeY * nNofChannels);
        if dataType != 6:  # samples are not XY
            for nAcqIndex in range(int(nNofAcquisitions)):
                # time stamps in file
                if 'ONOFF_ON' in S['TimestampState']:
                    timestamps[nAcqIndex] = np.fromfile(fid, np.float64, 1)
                # Skip preamble
                fid.seek(nPreambleInBytes, 1)
                # Get vertical values
                #   Decimate data by jumping decimationFactor samples
                # python doesn't support skip (decimationFactor-1) * sampleSizeY * nNofChannels
                yAux = np.fromfile(fid, np.dtype(dataTypeDict[dataType][3]), int(nSamples / bitpack * nNofChannels))

                # Use reshape to separate channels (undo the interleaving) -1 means as much as possible
                if bIsDigitalSource:
                    yCHs = np.unpackbits(yAux).astype(np.bool).reshape((-1, nNofChannels), order='C')  # reformatting
                else:
                    yCHs = yAux.reshape((-1, nNofChannels), order='C')  # Each row vector corresponds to one channel

                if dataType < 2:  # either int8 or int 16
                    # need to add the singel raw handling!
                    for idx in channelList:  # go through all channels independantly, but decrement before!
                        listIdx = channelList.index(idx)
                        y[:, nAcqIndex, listIdx] = yCHs[:, listIdx] * (
                                VerticalScale[idx] * VerticalDivisionCount / NofQuantisationLevels) + \
                                VerticalOffset[idx] # - (VerticalScale[idx] * VerticalPosition[idx])
                else:
                    # Assign values to the three dimensional array in case of multiple
                    # acquisitions and multiple channels
                    y[:, nAcqIndex, :] = yCHs

                # Skip postamble if this was not the last acquisition
                if nAcqIndex < nNofAcquisitions:
                    fid.seek(nPostambleInBytes, 1)
            # Reconstruct horizontal vector
            x = S['SignalResolution'] * np.arange((idxRange[0] - 1), idxRange[1], decimationFactor,
                                                  dtype=np.float64) + S['XStart']

        else:  # interleaved samples (XY)
            x = np.zeros((nSamples, nNofAcquisitions), dtype=np.float64)
            for nAcqIndex in range(int(nNofAcquisitions)):
                # time stamps in file
                if 'ONOFF_ON' in S['TimestampState']:
                    timestamps[nAcqIndex] = np.fromfile(fid, np.float64, 1)
                # Skip preamble
                fid.seek(nPreambleInBytes, 1)
                stride = dataTypeDict[dataType][0] + nNofChannels * dataTypeDict[dataType][1]
                # yPos = int(dataTypeDict[dataType][0] / dataTypeDict[dataType][1])
                yAux = fid.read(nSamples * stride)
                # fmtStr = 'd' + str(['f' for x in range(nNofChannels)])
                # yy = np.ndarray((nSamples,5),np.float32, yAux,0,(20,4),'C')[:,2:]
                # xx = np.ndarray((nSamples,1),np.float64, yAux,0,(20,8),'C')
                x[:, nAcqIndex] = np.ndarray((nSamples, 1), dtype=np.float64, buffer=yAux,
                                             strides=(stride, dataTypeDict[dataType][0]), order='C')[:, 0]
                # now truncate the first 8 byte wide x data
                y[:, nAcqIndex, :] = np.ndarray((nSamples, nNofChannels), dtype=np.float32, buffer=yAux,
                                                offset=dataTypeDict[dataType][0],
                                                strides=(stride, dataTypeDict[dataType][1]), order='C')
                # Skip postamble if this was not the last acquisition
                if nAcqIndex < nNofAcquisitions:
                    fid.seek(nPostambleInBytes, 1)

        # --- Close file ---
        fid.close()

    if 'ONOFF_ON' in S['TimestampState']:
        S[timestampkey] = timestamps.tolist()

    # --- Check horizontal length of y and x ---
    if len(y) != len(x):
        logger.warning('%s:NumelMismatch Sample vector y and horizontal vector x have different lenghts' % filename)

    return y, x, S


def is_num(s):
    try:
        float(s)
    except ValueError:
        return False
    else:
        return True
