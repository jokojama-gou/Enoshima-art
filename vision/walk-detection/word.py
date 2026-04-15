import zipfile
import re
import os
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

def adjust_word_total_time():
    # Tkinterの初期化と隠蔽
    root = tk.Tk()
    root.withdraw()

    # 1. ファイル選択ダイアログ
    file_path = filedialog.askopenfilename(
        title="編集時間を変更するWordファイルを選択してください",
        filetypes=[("Word files", "*.docx")]
    )

    if not file_path:
        return

    # 2. 編集時間の入力
    target_minutes = simpledialog.askinteger(
        "入力", 
        "設定したい総編集時間（分）を入力してください:",
        initialvalue=180,
        minvalue=0
    )

    if target_minutes is None:
        return

    backup_path = file_path + ".bak"
    new_file_path = file_path.replace(".docx", "_modified.docx")
    
    try:
        # バックアップの作成
        shutil.copy2(file_path, backup_path)
        
        # docx（zip）から app.xml を読み込んで置換
        with zipfile.ZipFile(file_path, 'r') as zin:
            app_xml_content = zin.read('docProps/app.xml').decode('utf-8')
        
        new_content = re.sub(
            r'(<TotalTime>)\d+(</TotalTime>)', 
            rf'\1{target_minutes}\2', 
            app_xml_content
        )
        
        # 新しいdocxの構築
        with zipfile.ZipFile(file_path, 'r') as zin:
            with zipfile.ZipFile(new_file_path, 'w') as zout:
                for item in zin.infolist():
                    if item.filename == 'docProps/app.xml':
                        zout.writestr(item, new_content)
                    else:
                        zout.writestr(item, zin.read(item.filename))
        
        messagebox.showinfo("完了", f"処理が成功しました。\n保存先: {new_file_path}\n総編集時間: {target_minutes}分")

    except Exception as e:
        messagebox.showerror("エラー", f"予期せぬ不具合が発生しました:\n{e}")
    
    finally:
        root.destroy()

if __name__ == "__main__":
    adjust_word_total_time()