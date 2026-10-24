# Troubleshooting: No Mobile Internet and APN Settings

Follow these steps in order when a customer has signal but mobile data does not work.

## Step 1 - Check balance and bundle

- Dial **\*250#** to confirm the line has remaining bundle data or balance.
- If the bundle is used up, data stops (Nile Flex) or is charged out of bundle (Nile Max). Suggest a data add-on.

## Step 2 - Basic checks on the phone

- Make sure Mobile data is switched on and Airplane mode is off.
- Toggle Airplane mode on and off, or restart the phone.
- Set the preferred network type to "5G/4G/3G auto" (or "4G/3G auto" on older phones).
- If the customer is abroad, Data roaming must be switched on and a roaming pass is recommended.

## Step 3 - APN settings

Most phones configure NileTel automatically. If not, create a new APN with these values:

| Field | Value |
|---|---|
| Name | NileTel Internet |
| APN | internet.niletel |
| Username / Password | (leave empty) |
| MCC / MNC | 602 / 99 |
| APN type | default,supl |
| APN protocol | IPv4/IPv6 |

- **Android:** Settings > Network & internet > SIMs > Access Point Names > Add (+).
- **iPhone:** Settings > Mobile Service > Mobile Data Network. iPhones normally receive the settings automatically after a restart.
- Customers can also send the word **APN** by SMS to 1250 to receive the settings automatically.

## Step 4 - Reset network settings

- Resetting network settings clears saved Wi-Fi passwords and APNs, then the APN is re-applied automatically.

## Step 5 - Still not working

- Try the SIM in another phone. If it fails there too, the SIM may be damaged (free 4G/5G SIM upgrade in stores).
- Otherwise open a network ticket with the customer's location; the target resolution time is 48 hours.
