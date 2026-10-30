'use client'

import { memo, useCallback, useEffect } from 'react'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import Chip from '@mui/material/Chip'
import Stack from '@mui/material/Stack'
import Tooltip from '@mui/material/Tooltip'
import CloudUploadIcon from '@mui/icons-material/CloudUpload'
import CloudDownloadIcon from '@mui/icons-material/CloudDownload'
import CopyAllIcon from '@mui/icons-material/CopyAll'
import PlayArrowIcon from '@mui/icons-material/PlayArrow'
import StopIcon from '@mui/icons-material/Stop'
import BoltIcon from '@mui/icons-material/Bolt'
import MemoryIcon from '@mui/icons-material/Memory'

import { useImageStore, useProcessingStore } from '@/services/stores'
import { copyImageToClipboard, downloadImage } from '@/services/utils'
import { detectHardwareCapabilities } from '@/services/inference'
import { useUpscale, useReducedMotion } from '@/hooks'
import { ProgressIndicator } from '@/components/feedback'
import { STYLES } from '@/constants'

interface ActionButtonsProps {
  onOpenModal: () => void
}

export const ActionButtons = memo(function ActionButtons({ onOpenModal }: ActionButtonsProps) {
  const outputURI = useImageStore((state) => state.outputURI)
  const status = useProcessingStore((state) => state.status)
  const activeProvider = useProcessingStore((state) => state.activeProvider)
  const deviceDetails = useProcessingStore((state) => state.deviceDetails)
  const setActiveProvider = useProcessingStore((state) => state.setActiveProvider)
  const prefersReducedMotion = useReducedMotion()

  const { execute, cancel } = useUpscale()

  // Detect GPU / WebGPU capabilities on mount if not yet detected
  useEffect(() => {
    if (activeProvider === 'unknown') {
      detectHardwareCapabilities().then(({ provider, details }) => {
        setActiveProvider(provider, details)
      })
    }
  }, [activeProvider, setActiveProvider])

  const isProcessing = status === 'processing'
  const hasOutput = outputURI !== null

  const showFlash = status === 'complete' && !prefersReducedMotion

  const handleDownload = useCallback(() => {
    if (outputURI) downloadImage(outputURI)
  }, [outputURI])

  const handleCopy = useCallback(() => {
    if (outputURI) copyImageToClipboard(outputURI)
  }, [outputURI])

  const isGpu = activeProvider === 'webgpu'
  const isWasm = activeProvider === 'wasm'

  const badgeLabel = isGpu
    ? deviceDetails && /rtx/i.test(deviceDetails)
      ? 'RTX GPU (WebGPU)'
      : 'GPU (WebGPU)'
    : isWasm
      ? 'CPU (WASM)'
      : 'Detecting...'

  const tooltipText = isGpu
    ? `Hardware acceleration active via WebGPU${deviceDetails ? ` (${deviceDetails})` : ''}`
    : isWasm
      ? 'Inference running on CPU via WebAssembly (WASM SIMD)'
      : 'Detecting hardware acceleration capabilities...'

  return (
    <Stack spacing={{ xs: 1.5, sm: 2 }}>
      <Button
        onClick={onOpenModal}
        startIcon={<CloudUploadIcon />}
        disabled={isProcessing}
        variant="contained"
        fullWidth
      >
        Choose Image
      </Button>

      {isProcessing && <ProgressIndicator compact />}

      {hasOutput ? (
        <>
          <Button
            onClick={handleDownload}
            startIcon={<CloudDownloadIcon />}
            variant="contained"
            fullWidth
            sx={showFlash ? STYLES.flashAnimation : undefined}
          >
            Download
          </Button>
          <Button
            onClick={handleCopy}
            startIcon={<CopyAllIcon />}
            variant="contained"
            fullWidth
          >
            Copy to Clipboard
          </Button>
        </>
      ) : (
        <Button
          onClick={isProcessing ? cancel : execute}
          startIcon={isProcessing ? <StopIcon /> : <PlayArrowIcon />}
          variant="contained"
          color={isProcessing ? 'error' : 'primary'}
          fullWidth
        >
          {isProcessing ? 'Cancel' : 'Run'}
        </Button>
      )}

      {/* Hardware Accelerator Status Badge */}
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', pt: 0.5 }}>
        <Tooltip title={tooltipText} arrow placement="top">
          <Chip
            size="small"
            icon={isGpu ? <BoltIcon fontSize="small" /> : <MemoryIcon fontSize="small" />}
            label={badgeLabel}
            variant={isGpu ? 'filled' : 'outlined'}
            color={isGpu ? 'primary' : 'default'}
            sx={{
              height: 24,
              fontSize: '0.72rem',
              fontWeight: 600,
              letterSpacing: '0.02em',
              cursor: 'default',
              '& .MuiChip-icon': {
                fontSize: '0.95rem',
              },
            }}
          />
        </Tooltip>
      </Box>
    </Stack>
  )
})

ActionButtons.displayName = 'ActionButtons'
