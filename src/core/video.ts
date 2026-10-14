import {
  createFile,
  DataStream,
  Endianness,
  type Sample,
  type MP4BoxBuffer,
} from "mp4box";
export interface Demuxed {
  config: VideoDecoderConfig;
  samples: Sample[];
  duration: number;
  width: number;
  height: number;
  origin: number;
}
export async function demux(file: File): Promise<Demuxed> {
  if (file.size > 300 * 1024 * 1024)
    throw Error(
      "This release accepts clips up to 300 MB. Trim the clip locally before importing.",
    );
  const buffer = (await file.arrayBuffer()) as MP4BoxBuffer;
  buffer.fileStart = 0;
  return new Promise((resolve, reject) => {
    const mp4 = createFile();
    let result: Demuxed | undefined;
    let expected = 0;
    let settled = false;
    mp4.onError = (...e: unknown[]) =>
      reject(Error(`Unable to read MP4/MOV container: ${e.join(" ")}`));
    mp4.onReady = (info) => {
      const track = info.videoTracks[0];
      if (!track?.video) {
        reject(Error("No video track found."));
        return;
      }
      const trak = mp4.getTrackById(track.id);
      const matrix = trak?.tkhd?.matrix;
      if (
        matrix &&
        (matrix[0] !== 65536 ||
          matrix[1] !== 0 ||
          matrix[3] !== 0 ||
          matrix[4] !== 65536)
      ) {
        reject(
          Error(
            "Rotated video is not supported by the analysis decoder yet. Export an upright MP4 locally first.",
          ),
        );
        return;
      }
      let description: Uint8Array | undefined;
      for (const entry of trak.mdia.minf.stbl.stsd.entries) {
        const e = entry as unknown as Record<
          string,
          { write: (s: DataStream) => void }
        >;
        const config = e.avcC || e.hvcC || e.vpcC || e.av1C;
        if (config) {
          const stream = new DataStream(undefined, 0, Endianness.BIG_ENDIAN);
          config.write(stream);
          description = new Uint8Array(stream.buffer, 8);
          break;
        }
      }
      expected = track.nb_samples;
      result = {
        config: {
          codec: track.codec,
          codedWidth: track.video.width,
          codedHeight: track.video.height,
          description,
        },
        samples: [],
        duration: track.duration / track.timescale,
        width: track.video.width,
        height: track.video.height,
        origin: 0,
      };
      mp4.setExtractionOptions(track.id, null, { nbSamples: 100 });
      mp4.start();
    };
    mp4.onSamples = (_id, _user, samples) => {
      if (!result) return;
      result.samples.push(...samples);
      if (result.samples.length >= expected && !settled) {
        settled = true;
        result.origin = Math.min(
          ...result.samples.map((s) => s.cts / s.timescale),
        );
        resolve(result);
      }
    };
    try {
      mp4.appendBuffer(buffer);
      mp4.flush();
      if (!settled)
        reject(
          Error(
            "Incomplete or unsupported video container. Export a standard H.264 MP4 locally.",
          ),
        );
    } catch (e) {
      reject(e);
    }
  });
}
export async function decodeVideo(
  file: File,
  visit: (frame: VideoFrame, time: number, meta: Demuxed) => Promise<void>,
  progress: (v: number) => void,
): Promise<Demuxed> {
  if (typeof VideoDecoder === "undefined")
    throw Error(
      "WebCodecs is unavailable. Use current desktop Chrome over HTTPS.",
    );
  const meta = await demux(file);
  const support = await VideoDecoder.isConfigSupported(meta.config);
  if (!support.supported)
    throw Error(
      `Chrome cannot decode ${meta.config.codec}. Convert to H.264 MP4 locally.`,
    );
  let chain = Promise.resolve(),
    pending = 0,
    error: unknown;
  let done = 0;
  const decoder = new VideoDecoder({
    output: (frame) => {
      pending++;
      chain = chain.then(async () => {
        try {
          if (!error)
            await visit(frame, frame.timestamp / 1e6 - meta.origin, meta);
        } catch (e) {
          error = e;
        } finally {
          frame.close();
          pending--;
          progress(++done / meta.samples.length);
        }
      });
    },
    error: (e) => {
      error = e;
    },
  });
  try {
    decoder.configure(meta.config);
    for (const sample of meta.samples) {
      if (error) throw error;
      while (decoder.decodeQueueSize > 3 || pending > 3) {
        await new Promise((r) => setTimeout(r, 2));
        if (error) throw error;
      }
      if (!sample.data) throw Error("Missing encoded sample data");
      decoder.decode(
        new EncodedVideoChunk({
          type: sample.is_sync ? "key" : "delta",
          timestamp: Math.round((sample.cts / sample.timescale) * 1e6),
          duration: Math.round((sample.duration / sample.timescale) * 1e6),
          data: sample.data,
        }),
      );
    }
    await decoder.flush();
    await chain;
    if (error) throw error;
    return meta;
  } finally {
    if (decoder.state !== "closed") decoder.close();
    await chain;
  }
}
