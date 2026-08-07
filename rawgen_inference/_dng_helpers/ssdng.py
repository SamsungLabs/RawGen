import warnings
import struct, imagecodecs
from tifffile import TiffFile
import  numpy as np
def read_eraw(dng_fn):
    image = TiffFile(dng_fn)
    image.series[0].keyframe.compression = 50002  # Need this for older versions of tifffile<2024.5.10
    image = image.series[0].keyframe.asarray()
    return image
def read_exif_header(file_path, offset):
    """ Read EXIF header up to a given offset and store it in a byte buffer. """
    buffer = bytearray()
    with open(file_path, "rb") as f:
        buffer.extend(f.read(offset))
    return buffer
def read_bytes_from_file(file_path, offset, length):
    """
    Reads a specified number of bytes from a file starting at a given offset.

    Args:
        file_path (str): Path to the file.
        offset (int): The byte offset where reading should begin.
        length (int): The number of bytes to read.

    Returns:
        bytes: The byte data read from the file.
    """
    try:
        # Open the file in binary mode ('rb')
        with open(file_path, 'rb') as file:
            # Move the file pointer to the given offset
            file.seek(offset)
            # Read the specified number of bytes
            byte_data = file.read(length)
        return byte_data
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
def append_image_to_buffer(buffer, image_path):
    """ Append an image (JPEG or JXL) to the buffer. """
    with open(image_path, "rb") as f:
        image_data = f.read()
    start_offset = len(buffer)
    buffer.extend(image_data)
    return start_offset, len(image_data)  # Return start offset and size
def append_byte_to_buffer(buffer, input_byte):
    """ Append an image (JPEG or JXL) to the buffer. """
    start_offset = len(buffer)  # The position where the image starts in buffer
    buffer.extend(input_byte)
    return start_offset, len(input_byte) 
def update_exif_tags(buffer, tag_offsets, values):
    """ Modify EXIF tags (little-endian format) in buffer at given offsets. """
    for offset, value in zip(tag_offsets, values):
        buffer[offset :offset+4] = struct.pack("<I", value)
def find_ss_jpeg_metadata(file_bytes, file_offset=0):
    marker = b'JKJK'
    marker_index = file_bytes.find(marker)
    if marker_index != -1:
        marker_index = marker_index + 4
        if (file_bytes[marker_index+3] & 0xe)  == 0xc:
            offset = 0x3a
        elif (file_bytes[marker_index+3] & 0xe) == 0xc:
            offset = 0x54
        else:
            return find_ss_jpeg_metadata(file_bytes[marker_index+1:], file_offset= file_offset + marker_index+1)
        # check for valid value
        asn_length = 3 * 4
        ccm_length = 9 * 4
        asn = file_bytes[marker_index + offset:marker_index + offset + asn_length]
        ccm = file_bytes[marker_index + offset+asn_length:marker_index + offset+asn_length + ccm_length]

        #  rewind 4 bytes to get full data
        marker_index = marker_index - 4
        full_binary_data = file_bytes[marker_index:marker_index + offset+asn_length + ccm_length + 4]
        return {'asn' : asn,
                'ccm': ccm,
                'full_data': full_binary_data,
                'marker_index': marker_index + file_offset}

def dump_ss_jpeg_metadata(byte_values):
    for i in range(0, len(byte_values), 4):
        value = struct.unpack('<i', byte_values[i:i+4])[0]
        value = value / (2 ** 16 - 1)
        print(f'value {i // 4}: ', value)

def save_to_dng(raw_recon, container_path, save_path,bitspersample, compression_type='jpegxl', recon_compression='jpegxl'):
    # raw_recon = np.round(raw_recon* (2 ** bitspersample - 1.0)).astype(np.uint16)
    raw_data_length_marker = bytes.fromhex('1701040001000000')
    raw_data_offset_marker = bytes.fromhex('1101040001000000')
    
    container_b = bytearray()
    
    with open(container_path, "rb") as f:
        container_b.extend(f.read())
    # kinda a trick here, the length and offset tags could be found in both the raw file and the embbed jpeg image,
    # however the tags for raw image are always placed before the ones for the jpeg image -> this find function will return the position of the tags for raw image.
    # we offset by 8 bytes to get the actual value stored in the tag
    length_idx = container_b.find(raw_data_length_marker) + 8
    offset_idx = container_b.find(raw_data_offset_marker) + 8
    output = bytearray()
    offset = struct.unpack('<I', container_b[offset_idx:offset_idx+4])[0]
    org_raw_len = struct.unpack('<I', container_b[length_idx:length_idx+4])[0]
    output.extend(container_b[:offset])
    if compression_type == 'jpegxl':
        jxl_data = imagecodecs.jpegxl_encode(raw_recon, lossless=True,bitspersample=bitspersample)
        output.extend(jxl_data)
    elif compression_type == 'jpeg':# and bitspersample == 12:
        assert raw_recon.dtype == np.uint16
        if recon_compression == 'jpegxl':
            jxl_data = imagecodecs.jpegxl_encode(raw_recon, lossless=True)
        elif recon_compression == 'uncompressed':
            jxl_data = raw_recon.flatten().tobytes()
        else:
            raise ValueError()
        output.extend(jxl_data)
        # we also need to read back the trailing info
        trailing = container_b[offset+org_raw_len:]
        output.extend(trailing)
        
        # update application note if exist
        application_note_marker = bytes.fromhex('bc020100')
        application_note_idx = container_b.find(application_note_marker)
        if (application_note_idx != -1):
            if (abs(application_note_idx - (offset_idx - 8)) % 12 != 0):
                warnings.warn("The application note position found might be wrong")
            application_note_idx = application_note_idx + 8
            app_note_pos = struct.unpack('<I', container_b[application_note_idx:application_note_idx+4])[0]
            if (app_note_pos > offset):
                app_note_pos = app_note_pos - org_raw_len + len(jxl_data)
                output[application_note_idx : application_note_idx + 4] = struct.pack('<I', app_note_pos)
        
        # update raw unique data if exist
        raw_unq_id_marker = bytes.fromhex('5dc6010010000000')
        raw_unq_id_idx = container_b.find(raw_unq_id_marker)
        if (raw_unq_id_idx != -1):
            raw_unq_id_idx = raw_unq_id_idx + 8
            raw_unq_id_pos = struct.unpack('<I', container_b[raw_unq_id_idx:raw_unq_id_idx+4])[0]
            if (raw_unq_id_pos > offset):
                raw_unq_id_pos = raw_unq_id_pos - org_raw_len + len(jxl_data)
                output[raw_unq_id_idx : raw_unq_id_idx + 4] = struct.pack('<I', raw_unq_id_pos)
        
        # update exif offset if exist:
        exif_offset_marker = bytes.fromhex('6987040001000000')
        exif_offset_idx = container_b.find(exif_offset_marker)
        if (exif_offset_idx != -1):
            exif_offset_idx = exif_offset_idx + 8
            exif_offset_pos = struct.unpack('<I', container_b[exif_offset_idx:exif_offset_idx+4])[0]
            if (exif_offset_pos > offset):
                exif_offset_pos = exif_offset_pos - org_raw_len + len(jxl_data)
                output[exif_offset_idx : exif_offset_idx + 4] = struct.pack('<I', exif_offset_pos)
                
                lens_value_marker = bytes.fromhex("34a40200")
                idx = container_b.find(lens_value_marker)
                if (idx != -1):
                    idx = idx + 8
                    pos = struct.unpack('<I', container_b[idx:idx+4])[0]
                    if (pos > offset):
                        pos = pos - org_raw_len + len(jxl_data)
                        if (idx > offset):
                            idx = idx - org_raw_len + len(jxl_data)
                        output[idx : idx + 4] = struct.pack('<I', pos)

        # update compression tag to jpegxl
        compression_marker = bytes.fromhex('0301030001000000')
        compression_idx = container_b.find(compression_marker)
        assert compression_idx != -1
        compression_idx  = compression_idx + 8
        if recon_compression == 'jpegxl':
            output[compression_idx : compression_idx + 4] = bytes.fromhex('42cd0000') 
        elif recon_compression == 'uncompressed':
            output[compression_idx : compression_idx + 4] = bytes.fromhex('01000000')
        else:
            raise ValueError()

        if recon_compression == 'uncompressed':    
            # update bits per sample
            bits_marker = bytes.fromhex('0c000c000c00') # 12 bits
            bits_idx = container_b.find(bits_marker)        
            assert bits_idx != -1        
            output[bits_idx : bits_idx + 6] = bytes.fromhex('100010001000') # change to 16 bits
    else:
        raise ValueError()
    

    output[length_idx : length_idx + 4] = struct.pack('<I', len(jxl_data))
    with open(save_path, 'wb') as f:
        f.write(output)
